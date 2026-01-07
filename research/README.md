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

### 1. Research Pipeline (Dataset Creation)

We provide a `Makefile` to simplify running the pipeline steps.

#### Using Make (Recommended)
Run these commands from the `research/` directory:
```bash
cd research

# Run full pipeline (Mine -> Generate -> Dataset)
make pipeline

# or run in test mode (faster)
make test-pipeline

# Run individual steps
make mine
make generate
```

#### Manual Execution

##### Mine DAGs from Airflow Repository
```bash
# Test mode (2 versions, quick validation)
python -m research.data.scripts.01_mine_dags --test

# Full mode (all versions from config)
python -m research.data.scripts.01_mine_dags

# Custom versions
python -m research.data.scripts.01_mine_dags --versions 3.0.0 3.0.1
```

#### Generate Instructions with Claude Batch API
```bash
# Test mode (5 DAGs)
python -m research.data.scripts.02_gen_instruct --test

# Full mode
python -m research.data.scripts.02_gen_instruct
```

**Note:** All research scripts must be run as modules using the `-m` flag from the project root directory. This ensures proper Python package resolution.

### 2. Research Notebooks

The project includes Jupyter notebooks for data analysis, fine-tuning, and evaluation:

#### For Google Colab (Fine-tuning)
Fine-tuning notebooks are designed for Google Colab with GPU support:
- `research/finetuning/notebooks/01_finetune.ipynb` - Model fine-tuning
- `research/finetuning/notebooks/02_generate_test_samples.ipynb` - Inference on test set

These notebooks include installation cells and will set up all dependencies automatically.

#### For Local Use (Analysis & Evaluation)
Some notebooks are designed for local execution:
- `research/data/analyse_tokens.ipynb` - Token distribution analysis
- `research/finetuning/notebooks/03_evaluate_generated_dags.ipynb` - DAG evaluation

**Local Setup:**
```bash
# Install with research dependencies
pip install -e ".[research]"

# Install Jupyter if not already available
pip install jupyter

# Launch Jupyter and ensure you select the venv kernel
jupyter notebook
```

**Important:** When running notebooks locally, make sure to select the correct Python kernel (the one from your virtual environment) in Jupyter/VSCode to ensure all imports work correctly.
