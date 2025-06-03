import json
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import os

# --- Configuration ---
MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct" # Used to load the correct tokenizer
MAX_SEQ_LENGTH = 2048 # Keep consistent with training script
SYNTHETIC_CSV_PATH = "./data/synthetic_financial_sentiment_english.csv"
TRAIN_JSONL_PATH = "./data/train_instructions.jsonl"
EVAL_JSONL_PATH = "./data/eval_instructions.jsonl"

SYSTEM_PROMPT = "You are a precise financial sentiment analysis assistant. Analyze the provided text and classify its sentiment as positive, negative, or neutral, according to the user's instruction."

# --- Helper Functions ---
def load_jsonl(file_path):
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_synthetic_csv(file_path):
    """Loads synthetic data CSV and converts to instruction format list."""
    df = pd.read_csv(file_path)
    # Ensure columns match the expected 'instruction', 'input', 'output'
    if not all(col in df.columns for col in ['instruction', 'input', 'output']):
        raise ValueError(f"Synthetic CSV must contain 'instruction', 'input', 'output' columns. Found: {df.columns}")
    return df.to_dict(orient='records')

def create_messages_format(sample, system_prompt=SYSTEM_PROMPT):
    """
    Creates the Llama-3 messages format for a single sample.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Ensure 'instruction', 'input', and 'output' keys exist in the sample
    if not all(k in sample for k in ['instruction', 'input', 'output']):
        raise KeyError(f"Sample is missing one of 'instruction', 'input', 'output'. Sample: {sample}")

    messages.extend([
        {"role": "user", "content": f"{sample['instruction']}\n\nText: {sample['input']}"},
        {"role": "assistant", "content": str(sample['output'])} # Ensure output is string
    ])
    return messages

# def formatting_func_for_sfttrainer(sample, tokenizer):
#     """
#     Formatting function for SFTTrainer.
#     Takes a sample (dict) and returns a string formatted using the chat template.
#     """
#     messages = create_messages_format(sample) # Uses global SYSTEM_PROMPT by default
#     formatted_string = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
#     return formatted_string

def apply_formatting_to_dataset(dataset, tokenizer_for_formatting):
    # This function now takes the tokenizer as an argument
    def _format_sample(sample):
        # This inner function matches the expected signature for .map()
        # and uses the tokenizer_for_formatting from the outer scope
        messages = create_messages_format(sample) # create_messages_format is from your script
        return {"text": tokenizer_for_formatting.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    return dataset.map(_format_sample, num_proc=4) # Adjust num_proc as needed

def get_prepared_datasets(tokenizer_name_or_path, synthetic_csv_p, train_jsonl_p, eval_jsonl_p):
    """
    Loads, combines, and prepares datasets for SFT.
    Returns:
        train_dataset_hf (Dataset): Formatted training Hugging Face Dataset.
        eval_dataset_hf (Dataset): Formatted evaluation Hugging Face Dataset.
        tokenizer (AutoTokenizer): Loaded tokenizer.
    """
    print(f"Loading tokenizer: {tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
    # Llama-3 specific tokenizer settings (Unsloth handles this well, but good to be explicit)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # It's important that padding_side is 'right' for decoder-only models during training
    # SFTTrainer/Unsloth usually handle this, but explicit for clarity if needed elsewhere.
    # tokenizer.padding_side = "right"


    print("--- Loading and Combining Datasets ---")
    # 1. Load original training data (English + Vietnamese)
    original_train_data = load_jsonl(train_jsonl_p)
    print(f"Loaded {len(original_train_data)} original training samples.") # Expected: 6000

    # 2. Load synthetic English training data
    synthetic_train_data = load_synthetic_csv(synthetic_csv_p)
    print(f"Loaded {len(synthetic_train_data)} synthetic training samples.") # Expected: 500

    # 3. Combine training data
    final_train_list = original_train_data + synthetic_train_data
    print(f"Total training samples: {len(final_train_list)}") # Expected: 6500

    # 4. Load evaluation data (English + Vietnamese)
    final_eval_list = load_jsonl(eval_jsonl_p)
    print(f"Total evaluation samples: {len(final_eval_list)}") # Expected: 500

    # 5. Convert to Hugging Face Dataset objects
    train_dataset_hf = Dataset.from_list(final_train_list)
    eval_dataset_hf = Dataset.from_list(final_eval_list)

    print("Datasets successfully loaded and combined.")
    return train_dataset_hf, eval_dataset_hf, tokenizer


if __name__ == "__main__":
    print("--- Part 2.1: Instruction Formatting ---")

    # Ensure data paths are correct
    if not os.path.exists(SYNTHETIC_CSV_PATH):
        raise FileNotFoundError(f"Synthetic data CSV not found at: {SYNTHETIC_CSV_PATH}")
    if not os.path.exists(TRAIN_JSONL_PATH):
        raise FileNotFoundError(f"Train JSONL not found at: {TRAIN_JSONL_PATH}")
    if not os.path.exists(EVAL_JSONL_PATH):
        raise FileNotFoundError(f"Eval JSONL not found at: {EVAL_JSONL_PATH}")

    train_hf, eval_hf, tokenizer_loaded = get_prepared_datasets(
        MODEL_NAME,
        SYNTHETIC_CSV_PATH,
        TRAIN_JSONL_PATH,
        EVAL_JSONL_PATH
    )

    print("\n--- Example Formatted Data ---")
    print(f"Loaded {len(train_hf)} training samples and {len(eval_hf)} evaluation samples.")

    num_examples_to_show = 3
    for i in range(min(num_examples_to_show, len(train_hf))):
        sample = train_hf[i]
        print(f"\n--- Example {i+1} ---")
        print("Original Sample (from dataset list):")
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Input: {sample['input']}")
        print(f"  Output: {sample['output']}")

        intermediate_messages = create_messages_format(sample)
        print("\nIntermediate 'messages' format:")
        for msg in intermediate_messages:
            print(f"  {msg}")

        final_templated_string = tokenizer_loaded.apply_chat_template(
            intermediate_messages, tokenize=False, add_generation_prompt=False
        )
        print("\nFinal form with chat template applied (showing special tokens):")
        # For very explicit special token display, one might need to decode token IDs one by one.
        # tokenizer.decode([token_id]) for token_id in tokenizer.encode(final_templated_string)
        # However, apply_chat_template with tokenize=False gives the string with special tokens like <|begin_of_text|>.
        print(final_templated_string)
        print("-" * 30)

    print("\nData formatting script finished. You can now import 'get_prepared_datasets' and 'formatting_func_for_sfttrainer' in your training script.")