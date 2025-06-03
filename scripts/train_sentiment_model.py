import os
from functools import partial
import torch
import wandb
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer

# Import functions from our formatting script
from format_data_for_sft import get_prepared_datasets, create_messages_format, MODEL_NAME, MAX_SEQ_LENGTH, SYNTHETIC_CSV_PATH, TRAIN_JSONL_PATH, EVAL_JSONL_PATH

# --- Configuration ---
# W&B Project Name
WANDB_PROJECT_NAME = "sea-fin-multilingual-sentiment"
# Hugging Face Hub Configuration
HF_MODEL_REPO_NAME = "jtz18/Llama-3.2-1B-Instruct-Financial-Sentiment-Multilingual" # Replace with YourHFUsername/RepoName
# Unsloth/Model Config
# MODEL_NAME and MAX_SEQ_LENGTH are imported
LOAD_IN_4BIT = True
DTYPE = None # None for auto detection (recommended by Unsloth)

# LoRA Configuration
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05 # Slightly increased dropout
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training Arguments
OUTPUT_DIR = "./results_llama3_2_1b_sentiment"
PER_DEVICE_TRAIN_BATCH_SIZE = 2 # Adjust based on your VRAM
PER_DEVICE_EVAL_BATCH_SIZE = 4  # Can be larger for eval
GRADIENT_ACCUMULATION_STEPS = 8 # Effective batch size = 2 * 8 = 16
WARMUP_RATIO = 0.05 # Use ratio for more flexibility with epochs
NUM_TRAIN_EPOCHS = 3 # Start with a few, can tune this
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
OPTIM = "adamw_8bit" # Unsloth recommends paged_adamw_8bit if 4bit, adamw_8bit otherwise
LR_SCHEDULER_TYPE = "cosine" # "linear" or "cosine" are common
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch" # Changed from "no" to "epoch" to evaluate during training
SEED = 42

def main():
    print("--- Part 2.2: Supervised Fine-Tuning ---")

    # --- 0. Setup W&B and Hugging Face Token ---
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT_NAME
    # Ensure you are logged in to Hugging Face CLI: `huggingface-cli login`
    # Or set HUGGING_FACE_HUB_TOKEN environment variable
    # wandb.login() # Call if you need to login programmatically

    print(f"Model will be pushed to: {HF_MODEL_REPO_NAME}")


    # --- 1. Load Datasets and Tokenizer ---
    # The tokenizer is loaded within get_prepared_datasets
    train_dataset, eval_dataset, tokenizer = get_prepared_datasets(
        MODEL_NAME,
        SYNTHETIC_CSV_PATH,
        TRAIN_JSONL_PATH,
        EVAL_JSONL_PATH
    )
    print(f"Loaded {len(train_dataset)} training samples and {len(eval_dataset)} evaluation samples.")

    # --- Pre-process datasets with formatting ---
    print("Applying formatting to datasets...")
    def _format_sample_for_map(sample, tokenizer_instance): # Renamed to avoid conflict if imported
        messages = create_messages_format(sample) # Ensure create_messages_format is accessible
        return {"text": tokenizer_instance.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
    map_function = partial(_format_sample_for_map, tokenizer_instance=tokenizer)

    formatted_train_dataset = train_dataset.map(map_function, num_proc=os.cpu_count() // 2 or 1)
    formatted_eval_dataset = eval_dataset.map(map_function, num_proc=os.cpu_count() // 2 or 1)
    print("Datasets formatted.")

    # --- 2. Load Model with Unsloth ---
    print(f"Loading base model: {MODEL_NAME}")
    model, _ = FastLanguageModel.from_pretrained( # Tokenizer already loaded, can ignore the one from here
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        token=os.getenv("HUGGING_FACE_HUB_TOKEN"), # Pass token if needed for gated models
    )
    print("Base model loaded.")

    # --- 3. Configure LoRA ---
    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth", # Recommended for Unsloth
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH, # Pass max_seq_length here as well
    )
    print("LoRA configured.")
    model.print_trainable_parameters()

    # --- 4. Define Training Arguments ---
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_ratio=WARMUP_RATIO,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        optim=OPTIM,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        seed=SEED,
        save_strategy=SAVE_STRATEGY,
        eval_strategy=EVAL_STRATEGY,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # or another metric like "accuracy", "f1", etc.
        greater_is_better=False,  # False for loss, True for metrics like accuracy
        # dataloader_num_workers = 2, # Adjust based on your system
    )

    # --- 5. Initialize SFTTrainer ---
    # The formatting_func needs the tokenizer, so we use a lambda or functools.partial
    # to pass it along with the sample.
    # SFTTrainer expects the formatting_func to take one argument (the sample)
    # or two if dataset_kwargs are provided.
    # So we need to wrap our formatting_func.

    # We can pass the tokenizer to the formatting_func via SFTTrainer's dataset_kwargs
    # if the func is designed to accept it, or use a lambda.
    # Since formatting_func_for_sfttrainer is defined as (sample, tokenizer),
    # SFTTrainer won't pass the tokenizer correctly by default.
    # The easiest is to make formatting_func_for_sfttrainer take only one argument (sample)
    # and have it access the tokenizer as a global or closure.
    # Let's adjust format_data_for_sft.py or use a lambda here.

    # Option 1: Modify formatting_func_for_sfttrainer to only take `sample` and use `tokenizer` from its scope
    # This is tricky if it's in another file unless tokenizer is global there.

    # Option 2: Use a lambda in SFTTrainer (cleanest for this structure)
    # sft_formatting_func = lambda sample: formatting_func_for_sfttrainer(sample, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_train_dataset,
        eval_dataset=formatted_eval_dataset,
        # formatting_func=sft_formatting_func, # Use the lambda
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
        dataset_text_field=None, # We use formatting_func
        packing=False, # Consider True for short sequences to speed up, but False is safer for varied lengths
    )
    print("SFTTrainer initialized.")

    # --- 6. Start Training ---
    print("Starting training...")
    try:
        train_result = trainer.train()
        print("Training finished.")

        # --- 7. Save Model and Metrics ---
        print("Saving final LoRA adapters...")
        final_adapter_path = os.path.join(OUTPUT_DIR, "final_lora_adapters")
        model.save_pretrained(final_adapter_path) # Saves LoRA adapters
        tokenizer.save_pretrained(final_adapter_path) # Save tokenizer with the adapters
        print(f"LoRA adapters saved to {final_adapter_path}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # --- 8. Push to Hugging Face Hub ---
        print(f"Pushing model and tokenizer to Hugging Face Hub: {HF_MODEL_REPO_NAME}")
        try:
            # For PEFT models, push_to_hub should handle adapters.
            # Unsloth's model object might have a specific method or work directly.
            model.push_to_hub(HF_MODEL_REPO_NAME, token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
            tokenizer.push_to_hub(HF_MODEL_REPO_NAME, token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
            print("Model and tokenizer pushed successfully.")
        except Exception as e:
            print(f"Error pushing to Hugging Face Hub: {e}")
            print("Please ensure you are logged in (`huggingface-cli login`) and have write permissions.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        # You might want to save the current state or adapters if an error occurs mid-training
        # model.save_pretrained(os.path.join(OUTPUT_DIR, "interrupted_lora_adapters"))
        # tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "interrupted_lora_adapters"))
    finally:
        wandb.finish()

if __name__ == "__main__":
    main()