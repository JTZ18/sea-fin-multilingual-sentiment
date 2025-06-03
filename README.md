# Multilingual Financial Sentiment Analysis LLM

This project focuses on building an end-to-end pipeline to fine-tune a Large Language Model (LLM) for multilingual financial sentiment analysis, specifically targeting English and Vietnamese. The goal is to identify the sentiment (positive, negative, or neutral) of financial news.

The project is structured to cover data exploration, data preparation (including synthetic data generation), model fine-tuning, and comprehensive model evaluation.

Access the project report here: [W&B Project Report](https://wandb.ai/jooz-cave/sea-fin-multilingual-sentiment/reports/SEA-FinTech-Sentiment-Analysis--VmlldzoxMzA3NDE2NQ).

Access the Finetuned Model here: [jtz18/Llama-3.2-1B-Instruct-Financial-Sentiment-Multilingual](https://huggingface.co/jtz18/Llama-3.2-1B-Instruct-Financial-Sentiment-Multilingual)

## Project Structure

```
.
├── data/                     # Stores processed datasets
│   ├── train_instructions.jsonl
│   ├── eval_instructions.jsonl
│   └── synthetic_financial_sentiment_english.csv
├── results_llama3_2_1b_sentiment/ # Stores fine-tuned model adapters
│   └── final_lora_adapters/
├── scripts/                  # Python scripts for the pipeline
│   ├── data_prep.py          # Script for cleaning and preparing datasets
│   ├── synthetic_data_generation.py # Script for generating synthetic English financial news
│   ├── format_data_for_sft.py # Script to format data for Supervised Fine-Tuning
│   ├── train_sentiment_model.py # Script for fine-tuning the LLM
│   └── evaluate_model.py     # Script for evaluating the fine-tuned model
├── task.ipynb                # Jupyter notebook for EDA, Tasks and Discussion Questions
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Pipeline Overview

The end-to-end pipeline consists of the following major stages:

1.  **Exploratory Data Analysis (EDA):**
    *   Analyze provided English (FinGPT/fingpt-sentiment-train) and Vietnamese (uitnlp/vietnamese_students_feedback) datasets.
    *   Identify data characteristics, class distributions, potential issues (duplicates, imbalances, short texts).
    *   Conclusions from EDA guide the data preparation steps.

2.  **Data Preparation (`scripts/data_prep.py`):**
    *   Cleans the English and Vietnamese datasets based on EDA findings.
    *   Maps sentiment labels to a unified set: "positive", "negative", "neutral".
    *   Standardizes the instruction format: `"What is the sentiment of this? Please choose an answer from {negative/neutral/positive}."`
    *   Selects 3,000 English and 3,000 Vietnamese samples for training, and 250 of each for evaluation, attempting stratified sampling.
    *   Outputs `train_instructions.jsonl` and `eval_instructions.jsonl`.

3.  **Synthetic Data Generation (`scripts/synthetic_data_generation.py`):**
    *   Uses `unsloth/Llama-3.2-1B-Instruct` to generate at least 500 synthetic English financial news snippets and their corresponding sentiment labels.
    *   Employs diverse meta-prompts for snippet generation and a constrained prompt for label generation.
    *   Includes automated cleaning and verification steps.
    *   Outputs `synthetic_financial_sentiment_english.csv`.

4.  **Instruction Formatting (`scripts/format_data_for_sft.py`):**
    *   Combines the original training data (6,000 samples) with the synthetic English data (500+ samples).
    *   Formats the combined training dataset (total ~6,500 samples) and the evaluation dataset (500 samples) into the Llama-3 chat message format, including a system prompt.
    *   Prepares data to be directly consumable by the `SFTTrainer`.

5.  **Supervised Fine-Tuning (SFT) (`scripts/train_sentiment_model.py`):**
    *   Fine-tunes the `unsloth/Llama-3.2-1B-Instruct` model using the prepared multilingual instruction dataset.
    *   Utilizes Unsloth for efficient training and PEFT (LoRA) to reduce computational requirements.
    *   Logs training metrics to Weights & Biases (W&B).
    *   Saves the fine-tuned LoRA adapters locally and pushes them to the Hugging Face Hub.
    *   **Base Model Chosen:** `unsloth/Llama-3.2-1B-Instruct` (for its instruction-following capabilities and efficient handling by Unsloth).

6.  **Model Evaluation (`scripts/evaluate_model.py`):**
    *   Evaluates the fine-tuned model against the original baseline model (`unsloth/Llama-3.2-1B-Instruct`).
    *   Uses the prepared evaluation dataset (250 English, 250 Vietnamese).
    *   Calculates standard classification metrics (accuracy, precision, recall, F1-score) overall, per language, and per class.
    *   Logs detailed results, including individual predictions and confusion matrices, to Weights & Biases.
    *   Leverages W&B Weave for interactive analysis of model performance.

## Setup and Installation

1.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

2.  **Install dependencies:**
    Ensure you have PyTorch installed with CUDA support if using a GPU. Then install other requirements:
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **Login to Services:**
    *   **Weights & Biases:**
        ```bash
        wandb login
        ```
    *   **Hugging Face Hub (for pushing models/adapters):**
        ```bash
        huggingface-cli login
        ```

## Running the Pipeline

Execute the scripts in the following order (note that the `task.ipynb` called the scripts in the following order):

1.  **Exploratory Data Analysis (Optional but Recommended):**
    Open and run the cells in `task.ipynb` using Jupyter Notebook or VS Code.

2.  **Data Preparation:**
    ```bash
    python scripts/data_prep.py
    ```
    This will create `data/train_instructions.jsonl` and `data/eval_instructions.jsonl`.

3.  **Synthetic Data Generation:**
    ```bash
    python scripts/synthetic_data_generation.py
    ```
    This will create `data/synthetic_financial_sentiment_english.csv`.

4.  **Instruction Formatting (usually called by `train_sentiment_model.py`):**
    The `scripts/format_data_for_sft.py` script contains helper functions used by the training script. It can also be run standalone to inspect formatted examples:
    ```bash
    python scripts/format_data_for_sft.py
    ```

5.  **Model Training:**
    Run the training script:
    ```bash
    python scripts/train_sentiment_model.py
    ```
    This will train the model, save adapters to `./results_llama3_2_1b_sentiment/final_lora_adapters`, and push to Hugging Face Hub. Training progress will be logged to W&B.

6.  **Model Evaluation:**
    Ensure your fine-tuned model adapters are available (either locally from the training step or accessible from Hugging Face Hub). The script prioritizes local adapters if found.
    ```bash
    python scripts/evaluate_model.py
    ```
    Evaluation results and detailed tables will be logged to W&B for analysis with Weave.

## Key Findings & Results

*   **EDA Insights:**
    *   The FinGPT English dataset had 9 sentiment classes, requiring mapping to 3 (positive, negative, neutral) to align with the Vietnamese dataset.
    *   Significant duplicates were found and handled in the English dataset.
    *   Both datasets exhibited class imbalances, which were addressed during sampling.
    *   The Vietnamese dataset had an additional "topic" label, not used in the final sentiment task.
*   **Model Performance:**
    *   The fine-tuned model (`jtz18/Llama-3.2-1B-Instruct-Financial-Sentiment-Multilingual`) significantly outperformed the baseline (`unsloth/Llama-3.2-1B-Instruct`).
    *   **Overall Accuracy:** Fine-tuned `0.824` vs. Baseline `0.547`.
    *   **English Accuracy:** Fine-tuned `0.804` vs. Baseline `0.538`.
    *   **Vietnamese Accuracy:** Fine-tuned `0.844` vs. Baseline `0.556`.
    *   The fine-tuned model showed excellent adherence to the output format (0 unparseable predictions compared to 8 for the baseline).
    *   Detailed metrics and confusion matrices are available on the associated [W&B Project](https://wandb.ai/jooz-cave/sea-fin-multilingual-sentiment/workspace?nw=nwuserjtz18).

## Discussion & Future Work

*   **Ensuring Fair Assessment:** Consistent prompting, standardized data, uniform parsing, and appropriate metric choices were key.
*   **Potential Improvements:**
    *   **Data:** Larger and more diverse datasets, advanced synthetic data generation, data augmentation.
    *   **Training:** Hyperparameter optimization, exploring different PEFT techniques or full fine-tuning.
    *   **Evaluation:** Expanded and challenge evaluation sets, human-in-the-loop evaluation, robustness checks.
*   **Future Exploration (Time/Compute Permitting):**
    *   Larger base models.
    *   Cross-lingual consistency checks.
    *   Explainability (XAI) methods.
    *   Multi-task learning for broader financial NLP capabilities.
    *   RLHF/DPO for further alignment.

## Weights & Biases Integration

This project heavily utilizes Weights & Biases for:
*   Tracking training experiments (loss, learning rate, custom metrics).
*   Logging evaluation results (accuracy, F1-scores, precision, recall).
*   Storing and versioning datasets and models (optional, used here for metrics).
*   Visualizing model performance with confusion matrices and custom charts.
*   Interactive analysis of evaluation data using W&B Weave.

Access the project report here: [W&B Project Report](https://wandb.ai/jooz-cave/sea-fin-multilingual-sentiment/reports/SEA-FinTech-Sentiment-Analysis--VmlldzoxMzA3NDE2NQ)

## License

[MIT](https://choosealicense.com/licenses/mit/)
