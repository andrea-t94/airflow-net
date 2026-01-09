# AirflowNet Research

This directory contains the experimental codebase for **AirflowNet**, focusing on mining Apache Airflow DAGs, creating instruction datasets, and fine-tuning Small Language Models (SLMs) to generate production-ready workflows.

## Directory Structure

- **`data/`**: Scripts and libraries for:
    - **Mining**: Extracting DAGs from GitHub repositories (`01_mine_dags.py`).
    - **Dataset Creation**: Transforming raw code into instruction pairs using techniques like Magpie (`02_gen_instruct.py`, `03_create_dataset.py`).
- **`finetuning/`**: Notebooks for training and evaluating models.
    - `notebooks/`: Sequential finetuning, inference, and evaluation steps.
- **`artifacts/`**:
    - `data/`: Intermediate datasets (mined DAGs, instructions).
    - `finetuning/`: Model inference outputs and evaluation results.
- **`lib/`**: Shared research utilities (e.g., `batch_processor.py`).

## Research Findings & Results

Our experiments focused on fine-tuning **Qwen 2.5 (1.5B/7B)** models on a curated dataset of Airflow DAGs. Below is a summary of our findings comparing the Fine-tuned model against the Base model:

### 1. Performance Improvements
- **Syntactic Validity**: The fine-tuned model demonstrates an approximate **8% reduction in invalid DAGs** (syntax errors, cyclic dependencies).
- **Modern Syntax Adoption**: A significant qualitative improvement is the adoption of modern Airflow features (e.g., TaskFlow API `@task` decorators) compared to the base model, which often defaults to deprecated operators.
- **Hallucination Control**: Reduced general hallucinations. The model adheres strictly to Airflow patterns, though it occasionally hallucinates internal testing libraries present in the training corpora (an area for future data cleaning).

### 2. Evaluation Strategy
We developed a robust "Judge" evaluation pipeline:
- **Parser Judge**: A static analysis tool that validates the Python code for Airflow-specific constraints.
- **LLM Judge**: Using **Claude 4.5 Sonnet** to grade generated code on:
    - **Correctness** (Logical flow)
    - **Completeness** (Imports, arguments)
    - **Best Practices** (Idiomatic Airflow usage)

### 3. Compute efficiency
- **Training**: Feasible on free-tier Colab (T4) for 1.5B models using Unsloth/QLoRA.
- **Inference**: High-throughput batch inference achievable on consumer hardware (e.g., Mac M1/M2/M3) using `llama.cpp` server with continuous batching.

## Reproducing the Research

If you want to recreate the dataset or run the research pipeline, follow these steps.

### 1. Environment Setup

The research pipeline requires additional dependencies (like `ipykernel`, `matplotlib`, `transformers`).

```bash
# Install research dependencies
uv sync --extra research
```

### 2. Research Pipeline (Dataset Creation)

We provide a `Makefile` to simplify running the pipeline steps (it automatically uses `uv run`).

#### Using Make (Recommended)
Run these commands from the `research/` directory:
```bash
cd research

# Run full pipeline (Mine -> Generate -> Dataset)
make pipeline

# or run in test mode (faster)
make test-pipeline
```

#### Manual Execution
If you prefer running manual commands, execute them from the **project root** using `uv run`.

**Mine DAGs:**
```bash
# Test mode
uv run -m research.data.scripts.01_mine_dags --test

# Full mode
uv run -m research.data.scripts.01_mine_dags
```

**Generate Instructions:**
```bash
# Test mode
uv run -m research.data.scripts.02_gen_instruct --test
```

### 3. Research Notebooks

The project includes Jupyter notebooks for data analysis, fine-tuning, and evaluation.

#### Running Notebooks

**1. Local Notebooks (CPU-friendly)**
Run these on your local machine using VS Code or Terminal:
- `research/data/analyse_tokens.ipynb`: Dataset analysis.
- `research/finetuning/notebooks/03_evaluate_generated_dags.ipynb`: Evaluation of generated DAGs.

**2. Cloud Notebooks (GPU Required)**
**IMPORTANT:** These notebooks are designed to be run on **Google Colab** (or a machine with a powerful GPU):
- `research/finetuning/notebooks/01_finetune.ipynb`: Fine-tuning the model.
- `research/finetuning/notebooks/02_generate_test_samples.ipynb`: Batch inference / Generation.
