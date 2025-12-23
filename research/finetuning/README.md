# Fine-tuning Module

This directory contains the notebooks and scripts used to fine-tune Large Language Models (specifically Qwen 2.5 Coding models) to generate high-quality Apache Airflow DAGs.

## Workflow

The process is divided into three sequential steps, implemented as Google Colab-ready notebooks in the `notebooks/` directory:

### 1. Fine-tuning (`01_finetune.ipynb`)
- **Objective**: Fine-tune a base model (e.g., Qwen 2.5 1.5B Instruct) on the dataset of Airflow DAGs.
- **Technique**: Uses **Unsloth** for efficient QLoRA fine-tuning.
- **Compute**: Optimized for Colab (runs on T4, faster on A100).
- **Output**: Saves the fine-tuned LoRA adapters and GGUF quantized models to Hugging Face or local artifacts.

### 2. Inference & Generation (`02_generate_test_samples.ipynb`)
- **Objective**: Generate samples from both the **Base Model** and the **Fine-tuned Model** using a hold-out test set.
- **Mechanism**: Runs local inference using `unsloth` (optimized for speed) with strict memory management to avoid OOM on T4 GPUs.
- **Output**: Produces JSONL files containing prompts, base model responses, and fine-tuned model responses.

### 3. Evaluation (`03_evaluate_generated_dags.ipynb`)
- **Objective**: Compare the quality of generated DAGs.
- **Metrics**:
    - **Parser-based**: Checks for syntactic validity, import errors, and graph cycles using a custom AST parser.
    - **LLM-based**: Uses **Claude 4.5 Sonnet** (via Anthropic Batch API) to qualitatively score DAGs on correctness, completeness, and adherence to best practices.
- **Output**: comprehensive CSV reports and visualizations demonstrating the model improvements.

## Usage

1. **Step 1**: Open `01_finetune.ipynb` in Colab. Configure your HF token and run to train the model.
2. **Step 2**: Open `02_generate_test_samples.ipynb` in Colab. Ensure it points to your newly trained model adapter. generated JSONL samples.
3. **Step 3**: Open `03_evaluate_generated_dags.ipynb`. Provide your Anthropic API key to run the qualitative evaluation and generate plots.
