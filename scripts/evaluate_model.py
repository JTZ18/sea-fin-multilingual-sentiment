import os
import json
import torch
import pandas as pd
import wandb
import re
from tqdm import tqdm

from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import from your existing SFT formatting script
# Ensure this script is in your Python path or the same directory
try:
    from format_data_for_sft import SYSTEM_PROMPT, create_messages_format, MODEL_NAME as BASE_MODEL_FOR_TOKENIZER
except ImportError:
    print("Error: Could not import from format_data_for_sft.py.")
    print("Please ensure format_data_for_sft.py is in the same directory or accessible in PYTHONPATH.")
    print("Using fallback SYSTEM_PROMPT and a placeholder for create_messages_format.")
    SYSTEM_PROMPT = "You are a precise financial sentiment analysis assistant. Analyze the provided text and classify its sentiment as positive, negative, or neutral, according to the user's instruction."
    BASE_MODEL_FOR_TOKENIZER = "unsloth/Llama-3.2-1B-Instruct" # Fallback
    def create_messages_format(sample, system_prompt=SYSTEM_PROMPT): # Fallback
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.extend([
            {"role": "user", "content": f"{sample['instruction']}\n\nText: {sample['input']}"},
            # For evaluation, 'output' is ground truth, not part of model input
        ])
        return messages


# --- Configuration ---
BASELINE_MODEL_NAME = "unsloth/Llama-3.2-1B-Instruct"
# Preference for FINETUNED_MODEL_PATH: Env Var > Local Path > Default HF Hub Path
# The training script saves adapters locally to "./results_llama3_2_1b_sentiment/final_lora_adapters"
LOCAL_FINETUNED_ADAPTERS_PATH = "./results_llama3_2_1b_sentiment/final_lora_adapters"
DEFAULT_HF_FINETUNED_REPO = "jtz18/Llama-3.2-1B-Instruct-Financial-Sentiment-Multilingual"

if os.getenv("FINETUNED_MODEL_PATH"):
    FINETUNED_MODEL_PATH = os.getenv("FINETUNED_MODEL_PATH")
elif os.path.exists(LOCAL_FINETUNED_ADAPTERS_PATH):
    FINETUNED_MODEL_PATH = LOCAL_FINETUNED_ADAPTERS_PATH
    print(f"Using local fine-tuned model adapters from: {FINETUNED_MODEL_PATH}")
else:
    FINETUNED_MODEL_PATH = DEFAULT_HF_FINETUNED_REPO
    print(f"Using fine-tuned model from Hugging Face Hub: {FINETUNED_MODEL_PATH}")
    print(f"Ensure you are logged in if this is a private model, or that the path is correct.")
    print(f"If you intended to use local adapters, ensure they exist at: {LOCAL_FINETUNED_ADAPTERS_PATH}")


EVAL_DATA_PATH = "./data/eval_instructions.jsonl"
MAX_SEQ_LENGTH = 2048  # Should be consistent with training
LOAD_IN_4BIT = True
DTYPE = None  # Auto-detection

WANDB_PROJECT_NAME = "sea-fin-multilingual-sentiment" # Same as training
WANDB_JOB_TYPE = "evaluation"

MAX_NEW_TOKENS_FOR_PREDICTION = 15 # Labels are short, give a bit of buffer
GENERATION_TEMPERATURE = 0.1 # For more deterministic output

# Define sentiment classes for metrics. Ensure order matches expectations if using 'labels' in sklearn.
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]


# --- Helper Functions ---

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_model_for_inference(model_name_or_path, is_peft_adapter=False, base_model_name_for_peft=None):
    print(f"Loading model: {model_name_or_path}")
    if is_peft_adapter:
        # For Unsloth, if model_name_or_path is an adapter directory,
        # it should load the base model specified in adapter_config.json and apply adapters.
        # The base model can also be explicitly passed if needed.
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path, # This should be the adapter path
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            # token = os.getenv("HUGGING_FACE_HUB_TOKEN"), # If model/adapters are private
            # base_model_name = base_model_name_for_peft # Unsloth usually infers this
        )
        print(f"Loaded PEFT model from adapters: {model_name_or_path}")
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name_or_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=DTYPE,
            load_in_4bit=LOAD_IN_4BIT,
            # token = os.getenv("HUGGING_FACE_HUB_TOKEN"), # If model is private
        )
        print(f"Loaded base model: {model_name_or_path}")

    # Ensure pad token is set for tokenizer (Llama specific)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For generation, padding side can be left or right, but left is often preferred for batching.
    # Unsloth handles this.

    return model, tokenizer

def format_evaluation_prompt_for_model(sample, tokenizer_for_template, system_prompt_text):
    """
    Formats a sample for evaluation inference.
    'output' in sample is ground truth, not used for prompt.
    """
    # Create a temporary sample without the 'output' for prompting,
    # or ensure create_messages_format handles it (it uses sample['output'] for assistant role).
    # For inference, we don't provide the assistant's turn.
    prompt_messages = []
    if system_prompt_text:
        prompt_messages.append({"role": "system", "content": system_prompt_text})
    prompt_messages.append({"role": "user", "content": f"{sample['instruction']}\n\nText: {sample['input']}"})

    return tokenizer_for_template.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True # CRUCIAL for instruct models to generate assistant response
    )

def get_prediction_from_model(model, tokenizer, prompt_text, max_new_toks, temp):
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH).to(model.device)

    with torch.no_grad(): # Ensure no gradients are computed
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_toks,
            temperature=temp,
            do_sample=True if temp > 0 else False, # Sample if temperature > 0
            pad_token_id=tokenizer.eos_token_id
        )
    # Decode only the newly generated tokens
    generated_text = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return generated_text.strip()

def parse_sentiment_from_response(response_text):
    response_lower = response_text.lower().strip()
    # More robust parsing: check for keywords as whole words or at start/end
    if re.search(r"\bpositive\b", response_lower): return "positive"
    if re.search(r"\bnegative\b", response_lower): return "negative"
    if re.search(r"\bneutral\b", response_lower): return "neutral"

    # Fallback for simple cases if model just outputs the word
    if "positive" in response_lower: return "positive" # Broader check
    if "negative" in response_lower: return "negative"
    if "neutral" in response_lower: return "neutral"

    # Try to extract first word if it's one of the labels
    first_word = response_lower.split(maxsplit=1)[0] if response_lower else ""
    first_word = re.sub(r'[^\w]', '', first_word) # remove punctuation
    if first_word in SENTIMENT_CLASSES:
        return first_word

    return "unparseable" # If no clear sentiment found

def calculate_and_log_metrics(df_subset, model_pred_col, true_label_col, class_labels_str, overall_log_prefix="", chart_title_suffix=""): # Added chart_title_suffix
    true_labels_series = df_subset[true_label_col]
    predicted_labels_series = df_subset[model_pred_col]

    # Filter out unparseable for classification_report and confusion_matrix
    parsable_indices = predicted_labels_series.isin(class_labels_str)
    parsable_true_str = true_labels_series[parsable_indices]
    parsable_pred_str = predicted_labels_series[parsable_indices]

    num_unparseable = len(df_subset) - len(parsable_pred_str)
    unparseable_rate = num_unparseable / len(df_subset) if len(df_subset) > 0 else 0

    wandb.log({f"{overall_log_prefix}unparseable_predictions_count": num_unparseable})
    wandb.log({f"{overall_log_prefix}unparseable_predictions_rate": unparseable_rate})

    if len(parsable_pred_str) == 0:
        print(f"Warning: No parsable predictions for {overall_log_prefix}. Skipping classification report and confusion matrix.")
        wandb.log({f"{overall_log_prefix}accuracy": 0.0})
        return

    accuracy = accuracy_score(true_labels_series, predicted_labels_series)
    report = classification_report(parsable_true_str, parsable_pred_str, labels=class_labels_str, output_dict=True, zero_division=0)

    label_to_int = {label: i for i, label in enumerate(class_labels_str)}
    parsable_true_int = parsable_true_str.map(label_to_int).tolist()
    parsable_pred_int = parsable_pred_str.map(label_to_int).tolist()

    wandb.log({f"{overall_log_prefix}accuracy": accuracy})
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            for metric_name, value in metrics.items():
                wandb.log({f"{overall_log_prefix}{label.replace(' ', '_')}_{metric_name}": value})
        elif label in ["accuracy"]:
             wandb.log({f"{overall_log_prefix}{label}": metrics})

    # Construct a more descriptive title for the confusion matrix
    cm_title = f"Confusion Matrix: {chart_title_suffix}" if chart_title_suffix else "Confusion Matrix"

    try:
        if not parsable_true_int or not parsable_pred_int:
            print(f"Not enough parsable data to plot confusion matrix for {overall_log_prefix}")
        else:
            wandb.log({f"{overall_log_prefix}confusion_matrix": wandb.plot.confusion_matrix(
                            preds=parsable_pred_int,
                            y_true=parsable_true_int,
                            class_names=class_labels_str,
                            title=cm_title  # Pass the custom title here
                        )})
    except Exception as e:
        print(f"Could not log confusion matrix for {overall_log_prefix} with title '{cm_title}': {e}")
        print(f"CM data: preds_int ({len(parsable_pred_int)}): {parsable_pred_int[:5]}, y_true_int ({len(parsable_true_int)}): {parsable_true_int[:5]}, class_names: {class_labels_str}")


def main():
    # --- 0. Initialize W&B ---
    wandb.init(project=WANDB_PROJECT_NAME, job_type=WANDB_JOB_TYPE, config={
        "baseline_model": BASELINE_MODEL_NAME,
        "finetuned_model": FINETUNED_MODEL_PATH,
        "eval_data": EVAL_DATA_PATH,
        "max_seq_length": MAX_SEQ_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS_FOR_PREDICTION,
        "generation_temperature": GENERATION_TEMPERATURE,
    })
    print(f"W&B Run: {wandb.run.get_url()}")


    # --- 1. Load Evaluation Data ---
    print(f"Loading evaluation data from: {EVAL_DATA_PATH}")
    eval_data_list = load_jsonl(EVAL_DATA_PATH)
    if not eval_data_list:
        print("Evaluation data is empty. Exiting.")
        wandb.finish()
        return
    print(f"Loaded {len(eval_data_list)} evaluation samples.")

    # --- 2. Load Models ---
    # Use BASE_MODEL_FOR_TOKENIZER for fine-tuned model if its tokenizer isn't found in adapter dir.
    # However, Unsloth's `from_pretrained` on adapter dir should also load its tokenizer.
    baseline_model, baseline_tokenizer = load_model_for_inference(BASELINE_MODEL_NAME)
    finetuned_model, finetuned_tokenizer = load_model_for_inference(FINETUNED_MODEL_PATH, is_peft_adapter=True)

    # --- 3. Run Inference ---
    evaluation_results = []
    print("Running inference on evaluation dataset...")

    for idx, sample in enumerate(tqdm(eval_data_list, desc="Evaluating Samples")):
        true_label = sample['output']
        lang = "EN" if idx < len(eval_data_list) / 2 else "VI" # Based on 250 EN / 250 VI structure

        # Baseline Model
        baseline_prompt = format_evaluation_prompt_for_model(sample, baseline_tokenizer, SYSTEM_PROMPT)
        baseline_raw_output = get_prediction_from_model(baseline_model, baseline_tokenizer, baseline_prompt, MAX_NEW_TOKENS_FOR_PREDICTION, GENERATION_TEMPERATURE)
        baseline_predicted_label = parse_sentiment_from_response(baseline_raw_output)

        # Fine-tuned Model
        finetuned_prompt = format_evaluation_prompt_for_model(sample, finetuned_tokenizer, SYSTEM_PROMPT)
        finetuned_raw_output = get_prediction_from_model(finetuned_model, finetuned_tokenizer, finetuned_prompt, MAX_NEW_TOKENS_FOR_PREDICTION, GENERATION_TEMPERATURE)
        finetuned_predicted_label = parse_sentiment_from_response(finetuned_raw_output)

        evaluation_results.append({
            "id": idx,
            "language": lang,
            "instruction": sample['instruction'],
            "input_text": sample['input'],
            "ground_truth": true_label,
            "baseline_prediction": baseline_predicted_label,
            "baseline_raw_output": baseline_raw_output,
            "baseline_correct": baseline_predicted_label == true_label if baseline_predicted_label != "unparseable" else False,
            "finetuned_prediction": finetuned_predicted_label,
            "finetuned_raw_output": finetuned_raw_output,
            "finetuned_correct": finetuned_predicted_label == true_label if finetuned_predicted_label != "unparseable" else False,
        })

    df_results = pd.DataFrame(evaluation_results)

    # --- 4. Calculate and Log Metrics ---
    print("\nCalculating and logging metrics...")

    # Overall Metrics
    print("\n--- Overall Metrics (Baseline) ---")
    calculate_and_log_metrics(df_results, "baseline_prediction", "ground_truth", SENTIMENT_CLASSES,
                            overall_log_prefix="overall_baseline_",
                            chart_title_suffix="Overall Baseline") # Add title suffix
    print("\n--- Overall Metrics (Fine-tuned) ---")
    calculate_and_log_metrics(df_results, "finetuned_prediction", "ground_truth", SENTIMENT_CLASSES,
                              overall_log_prefix="overall_finetuned_",
                              chart_title_suffix="Overall Fine-tuned") # Add title suffix

    # Per-Language Metrics
    df_en = df_results[df_results['language'] == 'EN']
    df_vi = df_results[df_results['language'] == 'VI']

    if not df_en.empty:
        print("\n--- English Metrics (Baseline) ---")
        calculate_and_log_metrics(df_en, "baseline_prediction", "ground_truth", SENTIMENT_CLASSES, "EN_baseline_")
        print("\n--- English Metrics (Fine-tuned) ---")
        calculate_and_log_metrics(df_en, "finetuned_prediction", "ground_truth", SENTIMENT_CLASSES, "EN_finetuned_")
    else:
        print("No English data found for per-language metrics.")

    if not df_vi.empty:
        print("\n--- Vietnamese Metrics (Baseline) ---")
        calculate_and_log_metrics(df_vi, "baseline_prediction", "ground_truth", SENTIMENT_CLASSES, "VI_baseline_")
        print("\n--- Vietnamese Metrics (Fine-tuned) ---")
        calculate_and_log_metrics(df_vi, "finetuned_prediction", "ground_truth", SENTIMENT_CLASSES, "VI_finetuned_")
    else:
        print("No Vietnamese data found for per-language metrics.")


    # --- 5. Log Detailed Results Table to W&B ---
    print("\nLogging detailed evaluation table to W&B...")
    wandb_table = wandb.Table(dataframe=df_results)
    wandb.log({"evaluation_details_table": wandb_table})

    # Save dataframe locally as well
    df_results.to_csv(os.path.join(wandb.run.dir, "evaluation_results_detailed.csv"), index=False)
    print(f"Detailed results saved locally to {os.path.join(wandb.run.dir, 'evaluation_results_detailed.csv')}")


    # --- 6. Finish W&B Run ---
    wandb.finish()
    print("Evaluation complete. Results logged to W&B.")
    print("You can now use W&B Weave to create dashboards from the 'evaluation_details_table'.")

if __name__ == "__main__":
    # Ensure necessary directories exist if saving anything locally by default
    # os.makedirs("./evaluation_outputs", exist_ok=True) # Example
    main()