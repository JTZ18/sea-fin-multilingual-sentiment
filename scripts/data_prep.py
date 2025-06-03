import pandas as pd
from datasets import load_dataset, concatenate_datasets
import re
import json

# --- Configuration ---
TARGET_INSTRUCTION = "What is the sentiment of this? Please choose an answer from {negative/neutral/positive}."
RANDOM_STATE = 42

# English Dataset Config
EN_TRAIN_SAMPLES = 3000
EN_EVAL_SAMPLES = 250
EN_MIN_WORD_COUNT = 3 # Filter for English text

# Vietnamese Dataset Config
VI_TRAIN_SAMPLES = 3000
VI_EVAL_SAMPLES = 250
VI_MIN_WORD_COUNT = 2 # Vietnamese sentences can be shorter

# --- Helper Functions ---
def map_english_sentiment(sentiment_str):
    positive_terms = ["positive", "moderately positive", "mildly positive", "strong positive"]
    negative_terms = ["negative", "moderately negative", "mildly negative", "strong negative"]
    if sentiment_str in positive_terms:
        return "positive"
    elif sentiment_str in negative_terms:
        return "negative"
    elif sentiment_str == "neutral":
        return "neutral"
    return None # Should not happen if data is clean

def map_vietnamese_sentiment(sentiment_int):
    if sentiment_int == 0: # Based on EDA: 0 maps to negative
        return "negative"
    elif sentiment_int == 1: # Based on EDA: 1 maps to neutral
        return "neutral"
    elif sentiment_int == 2: # Based on EDA: 2 maps to positive
        return "positive"
    return None # Should not happen

def clean_text_en(text):
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '', text)
    # Remove Twitter handles (optional, but common in financial tweets)
    text = re.sub(r'@\w+', '', text)
    # Remove stock tickers starting with $ (optional)
    text = re.sub(r'\$\w+', '', text)
    # Keep only alphanumeric and basic punctuation relevant to English
    # This helps remove emojis or characters from other languages if they are dominant
    # A more sophisticated language detection might be needed for mixed content,
    # but for now, we assume primary English.
    # text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', '', text) # This might be too aggressive
    text = text.strip()
    return text

def create_instruction_dataset(df, instruction_col_name='instruction', input_col_name='input', output_col_name='output'):
    """Converts a DataFrame to a list of dictionaries in instruction format."""
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "instruction": row[instruction_col_name],
            "input": row[input_col_name],
            "output": row[output_col_name]
        })
    return dataset

def export_to_jsonl(data, file_path):
    """Exports a list of dictionaries to a JSONL file."""
    import os
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# --- 1. English Dataset Preparation ---
print("--- Starting English Dataset Preparation ---")
# Load dataset
en_dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
df_en = en_dataset.to_pandas()
print(f"Original English dataset size: {len(df_en)}")

# Drop duplicates based on 'input'
df_en.drop_duplicates(subset=['input'], keep='first', inplace=True)
print(f"After dropping duplicates: {len(df_en)}")

# Clean 'input' text
df_en['input_cleaned'] = df_en['input'].apply(clean_text_en)

# Filter out empty or very short texts after cleaning
df_en = df_en[df_en['input_cleaned'].str.split().str.len() >= EN_MIN_WORD_COUNT]
# Filter out inputs that might be just symbols or non-english after cleaning (e.g. only ':)')
df_en = df_en[df_en['input_cleaned'].str.contains(r'[a-zA-Z]')] # Ensure some alphabetic chars
print(f"After text cleaning and length filtering: {len(df_en)}")

# Map sentiment labels to 3 classes ('positive', 'negative', 'neutral')
df_en['sentiment_mapped'] = df_en['output'].apply(map_english_sentiment)
df_en.dropna(subset=['sentiment_mapped'], inplace=True) # Remove rows where mapping failed
print(f"After sentiment mapping: {len(df_en)}")
print("\nEnglish sentiment distribution after mapping:")
print(df_en['sentiment_mapped'].value_counts(normalize=True))

# Add the standard instruction
df_en['instruction'] = TARGET_INSTRUCTION

# Select columns for the final dataset
df_en_processed = df_en[['instruction', 'input_cleaned', 'sentiment_mapped']].copy()
df_en_processed.rename(columns={'input_cleaned': 'input', 'sentiment_mapped': 'output'}, inplace=True)

# Stratified sampling for train and evaluation sets
# Ensure enough samples for eval first
n_per_class_eval_en = EN_EVAL_SAMPLES // 3
en_eval_list = []
remaining_en_list = []

for sentiment in df_en_processed['output'].unique():
    sentiment_subset = df_en_processed[df_en_processed['output'] == sentiment]
    if len(sentiment_subset) < n_per_class_eval_en:
        print(f"Warning: Not enough English samples for '{sentiment}' for evaluation. Taking all available: {len(sentiment_subset)}")
        eval_samples = sentiment_subset
    else:
        eval_samples = sentiment_subset.sample(n=n_per_class_eval_en, random_state=RANDOM_STATE)
    en_eval_list.append(eval_samples)
    remaining_en_list.append(sentiment_subset.drop(eval_samples.index))

df_en_eval_final = pd.concat(en_eval_list)
df_en_remaining = pd.concat(remaining_en_list)

# Adjust eval count if not perfectly divisible
if len(df_en_eval_final) < EN_EVAL_SAMPLES:
    needed = EN_EVAL_SAMPLES - len(df_en_eval_final)
    # Top up from the largest remaining classes
    top_up_samples = df_en_remaining.sample(n=needed, random_state=RANDOM_STATE) # Might not be stratified by sentiment
    df_en_eval_final = pd.concat([df_en_eval_final, top_up_samples])
    df_en_remaining = df_en_remaining.drop(top_up_samples.index)


n_per_class_train_en = EN_TRAIN_SAMPLES // 3
en_train_list = []
for sentiment in df_en_remaining['output'].unique():
    sentiment_subset = df_en_remaining[df_en_remaining['output'] == sentiment]
    if len(sentiment_subset) < n_per_class_train_en:
        print(f"Warning: Not enough English samples for '{sentiment}' for training after eval split. Taking all available: {len(sentiment_subset)}")
        train_samples = sentiment_subset
    else:
        train_samples = sentiment_subset.sample(n=n_per_class_train_en, random_state=RANDOM_STATE)
    en_train_list.append(train_samples)

df_en_train_final = pd.concat(en_train_list)

# Adjust train count if not perfectly divisible or if classes were small
if len(df_en_train_final) < EN_TRAIN_SAMPLES:
    needed = EN_TRAIN_SAMPLES - len(df_en_train_final)
    available_to_sample_more = df_en_remaining.drop(df_en_train_final.index)
    if len(available_to_sample_more) >= needed:
        top_up_samples = available_to_sample_more.sample(n=needed, random_state=RANDOM_STATE) # Might not be stratified by sentiment
        df_en_train_final = pd.concat([df_en_train_final, top_up_samples])
    else:
        print(f"Warning: Could not reach {EN_TRAIN_SAMPLES} for English training. Got {len(df_en_train_final)}")


print(f"\nFinal English training samples: {len(df_en_train_final)}")
print(df_en_train_final['output'].value_counts())
print(f"Final English evaluation samples: {len(df_en_eval_final)}")
print(df_en_eval_final['output'].value_counts())


# --- 2. Vietnamese Dataset Preparation ---
print("\n--- Starting Vietnamese Dataset Preparation ---")
# Load dataset splits
vi_train_ds = load_dataset("uitnlp/vietnamese_students_feedback", split="train")
vi_val_ds = load_dataset("uitnlp/vietnamese_students_feedback", split="validation")
vi_test_ds = load_dataset("uitnlp/vietnamese_students_feedback", split="test")

# Concatenate splits
vi_full_ds = concatenate_datasets([vi_train_ds, vi_val_ds, vi_test_ds])
df_vi = vi_full_ds.to_pandas()
print(f"Original Vietnamese dataset size (all splits): {len(df_vi)}")

# Remove the specific duplicate sentence that had conflicting sentiments
# "thầy dạy hay , tuy nhiên còn nhiều chỗ chưa thu hút ."
# This assumes 'sentence' is the column name for the text
duplicate_sentence_text = "thầy dạy hay , tuy nhiên còn nhiều chỗ chưa thu hút ."
# Get indices of all occurrences of this sentence
indices_to_drop = df_vi[df_vi['sentence'] == duplicate_sentence_text].index
if not indices_to_drop.empty:
    df_vi.drop(indices_to_drop, inplace=True)
    print(f"Removed {len(indices_to_drop)} instances of the problematic duplicate sentence.")
else:
    print("Problematic duplicate sentence not found (perhaps already handled or text differs slightly).")

# Drop other duplicates based on 'sentence', keeping the first
df_vi.drop_duplicates(subset=['sentence'], keep='first', inplace=True)
print(f"After dropping all duplicates: {len(df_vi)}")

# Filter by word count (Vietnamese sentences can be short but meaningful)
df_vi = df_vi[df_vi['sentence'].str.split().str.len() >= VI_MIN_WORD_COUNT]
print(f"After length filtering: {len(df_vi)}")

# Map sentiment labels
df_vi['sentiment_mapped'] = df_vi['sentiment'].apply(map_vietnamese_sentiment)
df_vi.dropna(subset=['sentiment_mapped'], inplace=True) # Should not drop any if mapping is exhaustive
print(f"After sentiment mapping: {len(df_vi)}")
print("\nVietnamese sentiment distribution after mapping:")
print(df_vi['sentiment_mapped'].value_counts(normalize=True))
print(df_vi['sentiment_mapped'].value_counts())


# Add the standard instruction
df_vi['instruction'] = TARGET_INSTRUCTION

# Select columns and rename
df_vi_processed = df_vi[['instruction', 'sentence', 'sentiment_mapped', 'topic']].copy()
df_vi_processed.rename(columns={'sentence': 'input', 'sentiment_mapped': 'output'}, inplace=True)

# Stratified sampling for train and evaluation sets for Vietnamese
# Special handling for Vietnamese 'neutral' class due to scarcity
vi_train_list = []
vi_eval_list = []
vi_remaining_df = df_vi_processed.copy() # Start with the full processed dataset

# Handle 'neutral' class first for training (take as many as possible up to a cap, or all)
neutral_vi_samples = vi_remaining_df[vi_remaining_df['output'] == 'neutral']
# We need VI_TRAIN_SAMPLES (3000) / 3 = 1000 ideally, VI_EVAL_SAMPLES (250) / 3 = ~83 ideally
target_neutral_train = VI_TRAIN_SAMPLES // 3
target_neutral_eval = VI_EVAL_SAMPLES // 3

# Take for eval first from neutral
if len(neutral_vi_samples) <= target_neutral_eval:
    neutral_eval = neutral_vi_samples.copy()
    neutral_train_available = pd.DataFrame(columns=neutral_vi_samples.columns) # Empty
else:
    neutral_eval = neutral_vi_samples.sample(n=target_neutral_eval, random_state=RANDOM_STATE)
    neutral_train_available = neutral_vi_samples.drop(neutral_eval.index)

vi_eval_list.append(neutral_eval)
vi_remaining_df = vi_remaining_df.drop(neutral_eval.index) # Remove selected eval samples

# Take for train from remaining neutral
if len(neutral_train_available) <= target_neutral_train:
    neutral_train = neutral_train_available.copy()
else:
    neutral_train = neutral_train_available.sample(n=target_neutral_train, random_state=RANDOM_STATE)

vi_train_list.append(neutral_train)
vi_remaining_df = vi_remaining_df.drop(neutral_train.index, errors='ignore') # Remove selected train samples

# Handle 'positive' and 'negative' classes
for sentiment in ['positive', 'negative']:
    sentiment_subset = vi_remaining_df[vi_remaining_df['output'] == sentiment]

    target_eval_count = VI_EVAL_SAMPLES // 3
    target_train_count = VI_TRAIN_SAMPLES // 3

    # If neutral eval took less, distribute the deficit to pos/neg
    if len(neutral_eval) < target_neutral_eval and len(vi_eval_list) < 3 : # only for first two iterations (pos/neg)
        deficit = target_neutral_eval - len(neutral_eval)
        target_eval_count += deficit // 2 # Distribute half of deficit

    if len(sentiment_subset) <= target_eval_count:
        eval_samples = sentiment_subset.copy()
        train_available = pd.DataFrame(columns=sentiment_subset.columns)
    else:
        eval_samples = sentiment_subset.sample(n=target_eval_count, random_state=RANDOM_STATE)
        train_available = sentiment_subset.drop(eval_samples.index)

    vi_eval_list.append(eval_samples)
    vi_remaining_df = vi_remaining_df.drop(eval_samples.index) # Update remaining

    # If neutral train took less, distribute the deficit
    if len(neutral_train) < target_neutral_train and len(vi_train_list) < 3:
         deficit = target_neutral_train - len(neutral_train)
         target_train_count += deficit // 2

    if len(train_available) <= target_train_count:
        train_samples = train_available.copy()
    else:
        train_samples = train_available.sample(n=target_train_count, random_state=RANDOM_STATE)
    vi_train_list.append(train_samples)
    vi_remaining_df = vi_remaining_df.drop(train_samples.index, errors='ignore') # Update remaining


df_vi_train_final = pd.concat(vi_train_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df_vi_eval_final = pd.concat(vi_eval_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

# Top-up if counts are low (usually for eval set if initial distribution was uneven)
if len(df_vi_eval_final) < VI_EVAL_SAMPLES:
    needed = VI_EVAL_SAMPLES - len(df_vi_eval_final)
    if len(vi_remaining_df) >= needed:
        top_up_samples = vi_remaining_df.sample(n=needed, random_state=RANDOM_STATE, replace=(len(vi_remaining_df) < needed))
        df_vi_eval_final = pd.concat([df_vi_eval_final, top_up_samples]).reset_index(drop=True)
        vi_remaining_df = vi_remaining_df.drop(top_up_samples.index, errors='ignore')
    else:
         print(f"Warning: Could not reach {VI_EVAL_SAMPLES} for Vietnamese eval. Got {len(df_vi_eval_final)}")


if len(df_vi_train_final) < VI_TRAIN_SAMPLES:
    needed = VI_TRAIN_SAMPLES - len(df_vi_train_final)
    if len(vi_remaining_df) >= needed:
        top_up_samples = vi_remaining_df.sample(n=needed, random_state=RANDOM_STATE, replace=(len(vi_remaining_df) < needed))
        df_vi_train_final = pd.concat([df_vi_train_final, top_up_samples]).reset_index(drop=True)
    else:
        print(f"Warning: Could not reach {VI_TRAIN_SAMPLES} for Vietnamese training. Got {len(df_vi_train_final)}")


# Ensure exact counts by trimming if over, or warning if under
if len(df_vi_train_final) > VI_TRAIN_SAMPLES:
    df_vi_train_final = df_vi_train_final.sample(n=VI_TRAIN_SAMPLES, random_state=RANDOM_STATE)
if len(df_vi_eval_final) > VI_EVAL_SAMPLES:
    df_vi_eval_final = df_vi_eval_final.sample(n=VI_EVAL_SAMPLES, random_state=RANDOM_STATE)


print(f"\nFinal Vietnamese training samples: {len(df_vi_train_final)}")
print(df_vi_train_final['output'].value_counts())
print(f"Final Vietnamese evaluation samples: {len(df_vi_eval_final)}")
print(df_vi_eval_final['output'].value_counts())
# The topic column is still in df_vi_train_final and df_vi_eval_final if needed for further analysis,
# but not part of the core instruction format here. We will select final columns.

df_en_train_final = df_en_train_final[['instruction', 'input', 'output']]
df_en_eval_final = df_en_eval_final[['instruction', 'input', 'output']]
df_vi_train_final = df_vi_train_final[['instruction', 'input', 'output']] # Removing 'topic' for final dataset
df_vi_eval_final = df_vi_eval_final[['instruction', 'input', 'output']]   # Removing 'topic' for final dataset


# --- 3. Combine and output (as instruction list format) ---
# For deliverable 1 & 2, these are the components from existing data
# (Synthetic data generation is a separate step not covered here)

english_train_instructions = create_instruction_dataset(df_en_train_final)
english_eval_instructions = create_instruction_dataset(df_en_eval_final)
vietnamese_train_instructions = create_instruction_dataset(df_vi_train_final)
vietnamese_eval_instructions = create_instruction_dataset(df_vi_eval_final)

print(f"\n--- Generated Datasets ---")
print(f"Number of English training instructions: {len(english_train_instructions)}")
print(f"Number of English evaluation instructions: {len(english_eval_instructions)}")
print(f"Number of Vietnamese training instructions: {len(vietnamese_train_instructions)}")
print(f"Number of Vietnamese evaluation instructions: {len(vietnamese_eval_instructions)}")

print("\nExample English Training Instruction:")
if english_train_instructions:
    print(english_train_instructions[0])
print("\nExample Vietnamese Training Instruction:")
if vietnamese_train_instructions:
    print(vietnamese_train_instructions[0])

# The deliverables want a single training dataset of 6500 (3k EN, 500 Synth EN, 3k VI)
# and a single eval dataset of 500 (250 EN, 250 VI).
# This script prepares the EN and VI components from existing data.
# The synthetic part would be added later.

# For now, let's show the combined train/eval from existing sources:
combined_training_instructions_from_source = english_train_instructions + vietnamese_train_instructions
combined_evaluation_instructions = english_eval_instructions + vietnamese_eval_instructions

print(f"\nTotal training instructions from source: {len(combined_training_instructions_from_source)}")
print(f"Total evaluation instructions from source: {len(combined_evaluation_instructions)}")

# --- 4. Export datasets to JSONL files ---
print("\n--- Exporting Datasets to JSONL ---")

# Export combined training instructions (without synthetic data for now)
export_to_jsonl(combined_training_instructions_from_source, "./data/train_instructions.jsonl")
print(f"Exported {len(combined_training_instructions_from_source)} training instructions to train_instructions.jsonl")

# Export combined evaluation instructions
export_to_jsonl(combined_evaluation_instructions, "./data/eval_instructions.jsonl")
print(f"Exported {len(combined_evaluation_instructions)} evaluation instructions to eval_instructions.jsonl")