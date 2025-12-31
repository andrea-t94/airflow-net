# AirflowNet
** The First Small Language Model (SML) Specialized for Apache Airflow**

AirflowNet is a research project dedicated to creating lightweight, specialized AI models capable of generating high-quality Apache Airflow DAGs. By fine-tuning efficient base models (like Qwen 2.5 1.5B) on a rigorously curated dataset, we aim to bring intelligent automation to data engineering workflows, running locally on consumer hardware.

---

## 🚀 Key Features

### 🧠 Specialized SML
-   **Model**: Fine-tuned **Qwen 2.5 1.5B Instruct**.
-   **Capabilities**: Generates syntactically correct, modern Airflow DAGs from natural language prompts.
-   **Efficiency**: Optimized for local inference (e.g., Mac M1/M2) using 4-bit quantization.

### 💎 Validated Data Pipeline
-   **Mining**: Extracts real-world usage patterns from thousands of heavily filtered DAGs.
-   **Synthetic Instructions**: Uses Claude 3.5 Sonnet (Batch API) to generate high-quality user intents.
-   **Result**: The `airflow-dag-dataset`, a gold-standard corpus for training code models.

### 🔌 Ecosystem Integration (WIP)
-   **MCP Server**: A Model Context Protocol server to provide LLMs with semantic search capabilities over the official Airflow codebase.
-   **CLI**: Unified `airflow-net` command for serving and inference.

---

## 📊 Project Status

| Component | Status | Notes |
| :--- | :--- | :--- |
| **Dataset Creation** | ✅ **Completed** | Mined, filtered, and synthetically augmented. |
| **Fine-tuning** | ✅ **Research Complete** | Model trained & validated (see [Research Summary](docs/research_summary.md)). |
| **Inference Engine** | 🚧 **In Beta** | Python `llama.cpp` server for high-throughput local serving. |
| **MCP Server** | 🚧 **In Development** | Indexing Airflow docs/code for RAG. |
| **VS Code Extension** | 📅 **Planned** | For seamless in-editor generation. |

---

## 📚 Documentation

-   [**01. Research Process**](docs/01_research_process.md): Deep dive into our findings, technical details on fine-tuning, and lessons learned.
-   [**02. Evaluation Methodology**](docs/02_evaluation_methodology.md): Structural and semantic evaluation criteria.
-   [**03. Inference & Benchmarks**](docs/03_inference_benchmarks.md): Hardware setup, attempts, and final performance numbers.
-   [**Changelog**](changelog.md): Track the evolution of the project.

---

## 🛠️ Usage

### Installation

We recommend using `uv` for lightning-fast dependency management, but standard `pip` works too.

#### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (optional, strictly recommended)

#### 1. Setup Virtual Environment
```bash
# Using uv (fastest)
uv venv
source .venv/bin/activate

# OR using standard python
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. Install Dependencies

**For Serving / Usage (Lightweight)**
If you just want to run the model server or use the CLI:
```bash
uv pip install -e .  # or pip install -e .
```

**For Research / Development**
If you want to run mining scripts, fine-tuning, or evaluation:
```bash
uv pip install -e ".[research]" # or pip install -e ".[research]"
```

### Serve the Model
```bash
# Start the local inference server (requires GGUF model)
airflow-net serve --model ./models/airflow-net-qwen2.5-1.5b.gguf
```

### Research Pipeline (For Dataset Creation)

If you want to recreate the dataset or run the research pipeline:

#### 1. Mine DAGs from Airflow Repository
```bash
# Test mode (2 versions, quick validation)
python -m research.data.scripts.01_mine_dags --test

# Full mode (all versions from config)
python -m research.data.scripts.01_mine_dags

# Custom versions
python -m research.data.scripts.01_mine_dags --versions 3.0.0 3.0.1
```

#### 2. Generate Instructions with Claude Batch API
```bash
# Test mode (5 DAGs)
python -m research.data.scripts.02_gen_instruct --test

# Full mode
python -m research.data.scripts.02_gen_instruct
```

**Note:** All research scripts must be run as modules using the `-m` flag from the project root directory. This ensures proper Python package resolution.

### Research Notebooks

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