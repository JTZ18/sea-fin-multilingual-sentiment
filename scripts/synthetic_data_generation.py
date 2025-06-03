import os
import torch
import pandas as pd
from tqdm import tqdm
import re
import random

from unsloth import FastLanguageModel
from transformers import TextStreamer

# --- Configuration ---
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct" # As per your script
MAX_SEQ_LENGTH = 2048
DTYPE = None  # None for auto detection. torch.float16 if an error occurs.
LOAD_IN_4BIT = True # Ensure this aligns with the model variant if it's 4-bit specific like bnb-4bit

NUM_SYNTHETIC_SAMPLES_TARGET = 500
# We'll aim to generate more raw snippets initially to account for losses during cleaning
# Your script was already generating NUM_SYNTHETIC_SAMPLES_TARGET + 300 unique snippets,
# which resulted in 800. We will now label all of these.
INITIAL_SNIPPET_GENERATION_TARGET = NUM_SYNTHETIC_SAMPLES_TARGET + 300 # Aim for 800 raw snippets

# This determines how many snippets are attempted per meta_prompt.
# With len(meta_prompts_for_snippets) and the outer loop break condition,
# this influences total generation attempts.
SNIPPETS_TO_GENERATE_PER_META_PROMPT = 5 # Reduced this as outer loop is more dominant

# For reproducibility
RANDOM_SEED = 42
if RANDOM_SEED:
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

OUTPUT_CSV_FILE = "./data/synthetic_financial_sentiment_english.csv"

# --- 1. Load Model and Tokenizer (Unsloth) ---
print(f"Loading model: {MODEL_NAME}")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT,
)
print("Model and tokenizer loaded.")

if torch.cuda.is_available():
    print("CUDA is available - Unsloth handles model placement for quantized models.")

# --- Helper Function for Text Generation ---
def generate_text_with_model(prompt_text, temperature=0.6, max_new_tokens=100, do_sample=True):
    inputs = tokenizer(prompt_text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return generated_text.strip()

# --- 2. Prompt Creation (Generating Diverse Financial News Snippets) ---
print("\n--- Step 2: Generating Diverse Financial News Snippets (Inputs) ---")

meta_prompts_for_snippets = [
    # Positive leaning
    "Generate a short, positive financial news headline about a company exceeding earnings expectations. Headline:",
    "Write a brief, optimistic news snippet about a successful new product launch in the tech sector. Snippet:",
    "Create a sentence describing a strong positive market trend for renewable energy stocks. Sentence:",
    "Draft a short news piece about a company announcing a significant stock buyback program. Piece:",
    "Generate a concise, upbeat report on a pharmaceutical company receiving FDA approval for a new drug. Report:",
    "Write a positive statement about a country's declining unemployment rate. Statement:",

    # Negative leaning
    "Generate a short, negative financial news headline about a company missing earnings estimates significantly. Headline:",
    "Write a brief, pessimistic news snippet about a major supply chain disruption affecting global trade. Snippet:",
    "Create a sentence describing a sharp decline in a specific commodity price. Sentence:",
    "Draft a short news piece about a company facing a major regulatory fine. Piece:",
    "Generate a concise, downbeat report on increasing inflation fears impacting consumer spending. Report:",
    "Write a negative statement about a company announcing unexpected layoffs. Statement:",

    # Neutral leaning
    "Generate a factual, neutral financial news headline about a routine CEO change at a large corporation. Headline:",
    "Write a brief, objective news snippet about the release of a government economic indicators report. Snippet:",
    "Create a sentence stating a financial fact about a company's market capitalization without expressing sentiment. Sentence:",
    "Draft a short news piece describing an upcoming merger between two mid-sized companies in the retail sector. Piece:",
    "Generate a concise, neutral report on a scheduled interest rate announcement by the central bank. Report:",
    "Write an objective statement about a company's quarterly revenue figures matching analyst consensus. Statement:",

    # Varying complexity / topic
    "Create a news blurb about the impact of new trade tariffs on the automotive industry. Blurb:",
    "Generate a short sentence about a shift in investor sentiment towards emerging markets. Sentence:",
    "Write a brief update on the performance of cryptocurrency markets today. Update:",
    "Draft a small news item about a new government infrastructure spending bill. Item:",

    # Market Sectors
    "Generate a short headline about recent volatility in the energy sector. Headline:",
    "Write a brief snippet about changing consumer trends affecting retail stocks. Snippet:",
    "Create a sentence about innovation driving growth in the healthcare sector. Sentence:",
    "Draft a news piece about declining profits in the banking sector. Piece:",
    "Generate a report on sustainability initiatives boosting agricultural stocks. Report:",

    # Economic Indicators
    "Write a headline about unexpectedly strong GDP growth figures. Headline:",
    "Create a snippet about housing market indicators showing signs of cooling. Snippet:",
    "Generate a sentence about manufacturing output declining for the third consecutive month. Sentence:",
    "Draft a news piece about consumer confidence reaching a five-year high. Piece:",
    "Write a report on rising producer price index affecting profit margins. Report:",

    # Corporate Events
    "Generate a headline about a surprise CEO resignation impacting stock price. Headline:",
    "Write a snippet about a major corporate restructuring plan announced today. Snippet:",
    "Create a sentence about a company spinning off its technology division. Sentence:",
    "Draft a news piece about a failed merger attempt between industry rivals. Piece:",
    "Generate a report on a successful IPO exceeding price expectations. Report:",

    # Global Finance
    "Write a headline about currency fluctuations affecting multinational corporations. Headline:",
    "Create a snippet about emerging market bonds facing increased selling pressure. Snippet:",
    "Generate a sentence about sovereign wealth funds increasing their technology investments. Sentence:",
    "Draft a news piece about international trade tensions affecting commodity prices. Piece:",
    "Write a report on central banks coordinating policy responses to economic uncertainty. Report:",

    # Investment Trends
    "Generate a headline about retail investors driving momentum in meme stocks. Headline:",
    "Write a snippet about institutional investors rotating from growth to value stocks. Snippet:",
    "Create a sentence about ESG investing becoming mainstream among pension funds. Sentence:",
    "Draft a news piece about algorithmic trading causing market irregularities. Piece:",
    "Generate a report on passive investment strategies outperforming active management. Report:"
]

generated_inputs_set = set() # Use a set for efficient uniqueness check
num_total_meta_prompts_iterations = (INITIAL_SNIPPET_GENERATION_TARGET // SNIPPETS_TO_GENERATE_PER_META_PROMPT) + len(meta_prompts_for_snippets)
# Ensure we iterate enough times by repeating meta_prompts if necessary
selected_meta_prompts_iterations = (meta_prompts_for_snippets * (num_total_meta_prompts_iterations // len(meta_prompts_for_snippets) + 1))[:num_total_meta_prompts_iterations]


print(f"Will iterate through meta prompts up to {len(selected_meta_prompts_iterations)} times, attempting {SNIPPETS_TO_GENERATE_PER_META_PROMPT} snippets per iteration.")
print(f"Targeting {INITIAL_SNIPPET_GENERATION_TARGET} unique raw snippets.")

for i, meta_prompt in enumerate(tqdm(selected_meta_prompts_iterations, desc="Generating Snippets")):
    if len(generated_inputs_set) >= INITIAL_SNIPPET_GENERATION_TARGET:
        print(f"Reached target of {INITIAL_SNIPPET_GENERATION_TARGET} unique snippets. Stopping snippet generation.")
        break

    for _ in range(SNIPPETS_TO_GENERATE_PER_META_PROMPT): # Try to get multiple snippets per meta-prompt
        if len(generated_inputs_set) >= INITIAL_SNIPPET_GENERATION_TARGET:
            break
        try:
            snippet = generate_text_with_model(
                meta_prompt,
                temperature=0.75,
                max_new_tokens=70,
                do_sample=True
            )
            if ":" in meta_prompt:
                last_part_of_meta = meta_prompt.split(":")[-1].strip()
                if snippet.startswith(last_part_of_meta) and len(last_part_of_meta) > 0:
                     snippet = snippet[len(last_part_of_meta):].strip()

            if snippet and len(snippet.split()) > 3:
                generated_inputs_set.add(snippet)
        except Exception as e:
            print(f"Error generating snippet for meta_prompt '{meta_prompt[:50]}...': {e}")
    if i % 50 == 0 and i > 0: # Log progress
        print(f"Progress: Iteration {i}, Unique snippets generated: {len(generated_inputs_set)}")


generated_inputs = sorted(list(generated_inputs_set))
print(f"Generated {len(generated_inputs)} unique potential input snippets.")

# --- 3. Generate Labels/Responses for ALL Snippets ---
print("\n--- Step 3: Generating Labels (Outputs) for ALL Snippets ---")

unified_instruction = "What is the sentiment of this? Please choose an answer from {negative/neutral/positive}."
synthetic_data_pairs = []

# Label ALL generated unique snippets
inputs_to_label = generated_inputs
print(f"Will label all {len(inputs_to_label)} unique generated snippets.")

for snippet in tqdm(inputs_to_label, desc="Generating Labels"):
    prompt_for_labeling = f"""Instruction: {unified_instruction}
News: {snippet}
Sentiment:"""

    try:
        label = generate_text_with_model(
            prompt_for_labeling,
            temperature=0.1,
            max_new_tokens=10,
            do_sample=True
        )
        label_cleaned = label.split()[0].lower().strip().replace('.', '').replace(',', '')
        synthetic_data_pairs.append({
            "instruction": unified_instruction,
            "input": snippet,
            "output_raw_llm": label,
            "output": label_cleaned
        })
    except Exception as e:
        print(f"Error generating label for snippet '{snippet[:50]}...': {e}")

print(f"Generated {len(synthetic_data_pairs)} initial instruction pairs (labels for all snippets).")

# --- 4. Cleaning, Verification, and FINAL Sampling ---
print("\n--- Step 4: Cleaning, Verification, and FINAL Sampling ---")
df_synthetic = pd.DataFrame(synthetic_data_pairs)

# a. Automated Checks
valid_labels = {"positive", "negative", "neutral"}
df_synthetic['label_is_valid'] = df_synthetic['output'].apply(lambda x: x in valid_labels)
invalid_label_count = len(df_synthetic[~df_synthetic['label_is_valid']])
print(f"Number of entries with invalid labels (before cleaning): {invalid_label_count}")

MIN_WORD_COUNT = 4
MAX_WORD_COUNT = 80
df_synthetic['word_count'] = df_synthetic['input'].apply(lambda x: len(x.split()))
df_synthetic['input_length_ok'] = (df_synthetic['word_count'] >= MIN_WORD_COUNT) & \
                                  (df_synthetic['word_count'] <= MAX_WORD_COUNT)
invalid_length_count = len(df_synthetic[~df_synthetic['input_length_ok']])
print(f"Number of entries with input length outside range ({MIN_WORD_COUNT}-{MAX_WORD_COUNT} words): {invalid_length_count}")

initial_rows = len(df_synthetic)
df_synthetic.drop_duplicates(subset=['input'], keep='first', inplace=True) # Should be minimal if inputs were unique
print(f"Removed {initial_rows - len(df_synthetic)} duplicate input snippets (post-labeling, if any).")

# Filter based on automated checks
df_cleaned = df_synthetic[df_synthetic['label_is_valid'] & df_synthetic['input_length_ok']].copy()
print(f"Number of entries after initial automated cleaning: {len(df_cleaned)}")

# Select final columns
df_final_synthetic = df_cleaned[['instruction', 'input', 'output']].copy()

# NOW, sample down to the target IF we have more than enough cleaned samples
if len(df_final_synthetic) > NUM_SYNTHETIC_SAMPLES_TARGET:
    print(f"We have {len(df_final_synthetic)} cleaned samples. Sampling down to {NUM_SYNTHETIC_SAMPLES_TARGET}.")
    # Stratified sampling to try and maintain label balance if possible
    if 'output' in df_final_synthetic.columns and len(df_final_synthetic['output'].unique()) > 1:
        try:
            # Calculate desired number of samples per class based on original proportions of the cleaned set
            # but scaled to NUM_SYNTHETIC_SAMPLES_TARGET
            current_proportions = df_final_synthetic['output'].value_counts(normalize=True)
            n_samples_per_class = (current_proportions * NUM_SYNTHETIC_SAMPLES_TARGET).round().astype(int)

            # Adjust sum to be exactly NUM_SYNTHETIC_SAMPLES_TARGET due to rounding
            diff = NUM_SYNTHETIC_SAMPLES_TARGET - n_samples_per_class.sum()
            if diff != 0:
                # Add/subtract difference to the class with most/least samples or prioritize largest proportion
                # This simple adjustment adds to the class that has the largest count in n_samples_per_class
                if diff > 0:
                    for _ in range(diff):
                        n_samples_per_class[n_samples_per_class.idxmax()] += 1
                else: # diff < 0
                     for _ in range(abs(diff)):
                        n_samples_per_class[n_samples_per_class.idxmin()] -=1
                        if n_samples_per_class[n_samples_per_class.idxmin()] < 0 : # safety
                            n_samples_per_class[n_samples_per_class.idxmin()] = 0


            # Ensure no class requests more samples than available after cleaning
            temp_df_list = []
            for sentiment_class, n_needed in n_samples_per_class.items():
                class_df = df_final_synthetic[df_final_synthetic['output'] == sentiment_class]
                n_available = len(class_df)
                n_to_sample = min(n_needed, n_available)
                if n_to_sample > 0 : # only sample if n_to_sample is positive
                    temp_df_list.append(class_df.sample(n=n_to_sample, random_state=RANDOM_SEED if RANDOM_SEED else None))

            df_final_synthetic_sampled = pd.concat(temp_df_list)

            # If total samples are still not NUM_SYNTHETIC_SAMPLES_TARGET (e.g. due to min constraint),
            # top up or trim with random sampling from the combined stratified sample or original cleaned set
            if len(df_final_synthetic_sampled) < NUM_SYNTHETIC_SAMPLES_TARGET and len(df_final_synthetic) >= NUM_SYNTHETIC_SAMPLES_TARGET :
                needed_more = NUM_SYNTHETIC_SAMPLES_TARGET - len(df_final_synthetic_sampled)
                # Get remaining items from df_final_synthetic not already in df_final_synthetic_sampled
                remaining_df = df_final_synthetic.loc[~df_final_synthetic.index.isin(df_final_synthetic_sampled.index)]
                if len(remaining_df) >= needed_more:
                     df_final_synthetic_sampled = pd.concat([df_final_synthetic_sampled, remaining_df.sample(n=needed_more, random_state=RANDOM_SEED if RANDOM_SEED else None)])
            elif len(df_final_synthetic_sampled) > NUM_SYNTHETIC_SAMPLES_TARGET:
                df_final_synthetic_sampled = df_final_synthetic_sampled.sample(n=NUM_SYNTHETIC_SAMPLES_TARGET, random_state=RANDOM_SEED if RANDOM_SEED else None)

            df_final_synthetic = df_final_synthetic_sampled.sample(frac=1, random_state=RANDOM_SEED if RANDOM_SEED else None).reset_index(drop=True) # shuffle
            print(f"Sampled down to {len(df_final_synthetic)} using stratified sampling (or best effort).")

        except Exception as e:
            print(f"Stratified sampling failed: {e}. Falling back to random sampling.")
            df_final_synthetic = df_final_synthetic.sample(n=NUM_SYNTHETIC_SAMPLES_TARGET,
                                                           random_state=RANDOM_SEED if RANDOM_SEED else None)
    else: # Fallback if only one class or no 'output' column, or not enough unique classes for stratification
         df_final_synthetic = df_final_synthetic.sample(n=NUM_SYNTHETIC_SAMPLES_TARGET,
                                                           random_state=RANDOM_SEED if RANDOM_SEED else None)

elif len(df_final_synthetic) < NUM_SYNTHETIC_SAMPLES_TARGET:
    print(f"WARNING: After cleaning, we have {len(df_final_synthetic)} samples, which is less than the target {NUM_SYNTHETIC_SAMPLES_TARGET}.")
    print("Consider increasing INITIAL_SNIPPET_GENERATION_TARGET, relaxing cleaning criteria, or improving label generation if appropriate.")
# If len(df_final_synthetic) == NUM_SYNTHETIC_SAMPLES_TARGET, we do nothing.

print(f"Final number of synthetic samples: {len(df_final_synthetic)}")

# b. Human-in-the-Loop Validation (Placeholder - IMPORTANT!)
print("\n--- Human-in-the-Loop Validation (Manual Step) ---")
print("The generated dataset (df_final_synthetic) should now be manually reviewed.")
# (Rest of the human validation print statements are good)

# --- 5. Save the Synthetic Dataset ---
print("\n--- Step 5: Saving the Synthetic Dataset ---")
if not df_final_synthetic.empty:
    df_final_synthetic.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Synthetic dataset saved to {OUTPUT_CSV_FILE}")
else:
    print("No data to save, df_final_synthetic is empty.")


# Display some info about the final dataset
print("\n--- Final Dataset Information ---")
print(f"Total samples: {len(df_final_synthetic)}")
if not df_final_synthetic.empty:
    print("Sentiment Distribution:")
    print(df_final_synthetic['output'].value_counts(normalize=True))
    print("\nFirst 5 samples:")
    print(df_final_synthetic.head())
else:
    print("No data generated or all data filtered out.")

# (Documentation notes are good)
print("\n--- Documentation Notes ---")
print(f"Model used for generation: {MODEL_NAME}")
print(f"Target number of samples: {NUM_SYNTHETIC_SAMPLES_TARGET}")
print(f"Initial raw snippets targeted before labeling: {INITIAL_SNIPPET_GENERATION_TARGET} (actually generated: {len(inputs_to_label)})")
print("Snippet Generation Strategy:")
print("  - Used a list of diverse 'meta_prompts' to guide the LLM.")
print("  - LLM prompted to complete these meta_prompts (e.g., 'Headline:', 'Snippet:').")
print("  - Temperature for snippet generation: 0.75 (for diversity).")
print("Label Generation Strategy:")
print(f"  - Used the unified instruction: '{unified_instruction}'")
print("  - Formatted prompt: Instruction + News Snippet + 'Sentiment:' marker.")
print("  - Temperature for label generation: 0.1 (for consistency).")
print("  - Labels cleaned to be one of: 'positive', 'negative', 'neutral'.")
print("Validation Steps:")
print("  - Automated: Checked label validity, input snippet word count, removed duplicates.")
print("  - Cleaned data pool was then sampled down to target if overage existed.")
print("  - Manual Review: CRITICAL. The generated data needs human oversight for quality and accuracy.")
print(f"Random Seed for generation (if used): {RANDOM_SEED}")
print("To ensure desired format for labels:")
print("  1. Explicit instruction: '...Please choose an answer from {negative/neutral/positive}'.")
print("  2. Used an 'Instruct' model variant.")
print("  3. Low temperature (0.1) for label generation.")
print("  4. `max_new_tokens` set to a small value (10) for labels.")
print("  5. Post-generation cleaning of the label string (e.g., .split()[0].lower()).")