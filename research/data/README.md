# AirflowNet Data Pipeline

This directory contains the tools and scripts for creating the **AirflowNet** training dataset. The pipeline consists of three sequential steps: mining raw DAGs from Apache Airflow repositories, generating instruction-following pairs using Claude, and formatting/uploading the final dataset.

## 📂 Directory Structure

```
research/data/
├── config/             # Configuration files
│   ├── mining_config.yaml
│   └── generation_config.yaml
├── lib/                # Shared library code
│   ├── mining.py       # GitHub mining logic
│   ├── instruction.py  # Claude API interaction
│   └── config_loader.py
├── scripts/            # Executable pipeline scripts
│   ├── 01_mine_dags.py
│   ├── 02_gen_instruct.py
│   └── 03_create_dataset.py
└── artifacts/      # Research artifacts (gitignored)
    ├── 01_raw_dags/
    ├── 02_instruct_dags/
    ├── magpie_data/
    └── inference/
```

## 🛠️ Prerequisites

Ensure you have the project dependencies installed:
```bash
pip install -e .
```

You will need the following environment variables in a `.env` file at the project root:
```ini
GITHUB_TOKEN=your_github_token          # For Step 1 (Mining)
ANTHROPIC_API_KEY=your_anthropic_key    # For Step 2 (Generation)
HF_TOKEN=your_huggingface_token         # For Step 3 (Upload)
```

---

## 🚀 Pipeline Steps

### Step 1: Mining Raw DAGs
**Script**: `scripts/01_mine_dags.py`

Fetches `example_*.py` DAGs from the official Apache Airflow repository across multiple versions defined in `config/mining_config.yaml`.

- **Input**: GitHub API (Apache Airflow Repo)
- **Output**: `research/artifacts/01_raw_dags/dags.jsonl`
- **Features**:
  - Validates DAG syntax using AST.
  - Extracts metadata (task count, dependencies).
  - Skips non-DAG Python files.

**Usage**:
```bash
# Run full mining process
python research/data/scripts/01_mine_dags.py
```

### Step 2: Instruction Generation
**Script**: `scripts/02_gen_instruct.py`

Uses Anthropic's Claude API (via Message Batches) to analyze each DAG and generate high-quality natural language instructions (e.g., "Create a DAG that runs every daily and extracts data from S3").

- **Input**: `research/artifacts/01_raw_dags/dags.jsonl`
- **Output**: `research/artifacts/02_instruct_dags/airflow_instructions.jsonl`
- **Features**:
  - **Filters out invalid DAGs** (syntax errors, circular dependencies).
  - Generates multiple instruction variants per DAG.
  - Formats data for ChatML (System/User/Assistant).

**Usage**:
```bash
# Generate instructions for all mined DAGs
python research/data/scripts/02_gen_instruct.py

# Run a test batch (limit to 5 DAGs)
python research/data/scripts/02_gen_instruct.py --test
```

### Step 3: Dataset Creation & Upload
**Script**: `scripts/03_create_dataset.py`

Validates, splits, and uploads the final dataset to the HuggingFace Hub.

- **Input**: `research/artifacts/02_instruct_dags/airflow_instructions.jsonl`
- **Output**: HuggingFace Dataset
- **Features**:
  - Validates ChatML format.
  - Performs train/test split.
  - Uploads to the configured repository.

**Usage**:
python research/data/scripts/03_create_dataset.py
```
