## Repository Overview

* **Root Philosophy:** "Library First." The core logic is a Python package (`src/airflow_net`) that handles generation and validation.
* **Distribution:** Users install the package to run the agent locally via CLI, MCP (Claude), or HTTP (Cursor).
* **Research:** All data mining and training logic lives in `research/` and *imports* the core library to ensure the validator used for training is identical to the one used for inference.

---

## Directory Tree

```text
airflow-net/
├── pyproject.toml              # main dependency definition (for the library)
├── README.md                   # Documentation for the library
├── Makefile                    # Shortcuts: install, test, serve, research-setup
│
├── src/                        # 📦 THE PRODUCT (Pip installable package)
│   └── airflow_net/
│       ├── __init__.py         # Exports AirflowAgent, DAGValidator
│       ├── agent.py            # The "Loop": Orchestrates Generation -> Validation -> Retry
│       ├── engine.py           # The "Muscle": Wraps llama.cpp & Hardware Detection
│       ├── validation.py       # The "Brain": Pure logic AST validator (Refactored from lib/dag_parser.py)
│       ├── cli.py              # Entry point for `airflow-net` command
│       ├── utils.py            # Shared utilities (logging, config loading)
│       └── interfaces/         # 🔌 Connectors for external tools
│           ├── __init__.py
│           ├── mcp.py          # Model Context Protocol server (for Claude Desktop)
│           └── http.py         # OpenAI-compatible API server (for Cursor/VS Code)
│
├── research/                   # 🏭 THE FACTORY (Data creation & Training)
│   ├── requirements.txt        # Shared dependencies
│   │
│   ├── data/                   # Step 1: Dataset Creation
│   │   ├── config/             # Configuration for mining/generation
│   │   │   ├── mining_config.yaml
│   │   │   └── generation_config.yaml
│   │   ├── lib/                # Data generation logic
│   │   │   ├── mining.py
│   │   │   ├── instruction.py
│   │   │   └── config_loader.py    # Configuration and environment utilities
│   │   └── scripts/            # Workflow steps
│   │       ├── 01_mine_dags.py
│   │       ├── 02_gen_instruct.py
│   │       └── 03_upload_hf.py
│   │
│   └── finetuning/             # Step 2: Model Training
│       └── notebooks/
│           ├── colab_inference.ipynb
│           ├── model_evaluation.ipynb
│           └── finetune.ipynb
│
└── tests/                      # Unit tests
    └── test_validation.py      # Critical: Ensure DAGValidator catches known bad syntax

```

---

## Component Details

### 1. `src/airflow_net` (The Library)

*Dependencies: `llama-cpp-python`, `pydantic`, `click`, `mcp`.*

* **`validation.py`**:
* **Source:** Refactored from `lib/dag_parser.py`.
* **Changes:** Removed file I/O. Now accepts a string (`code_content`) and returns a list of `ValidationError` objects. This is the "Shared Kernel."


* **`engine.py`**:
* **Source:** Adapted from `scripts/dag_generation_llamacpp.py`.
* **Responsibilities:**
* `ModelEngine` class: Handles loading GGUF models.
* **Hardware Detection:** Automatically sets `n_gpu_layers` based on `torch.cuda.is_available()` or MPS (Mac) checks, replacing the old `.sh` scripts.




* **`agent.py`**:
* **New Component.**
* **Logic:** Implements the **Self-Correction Loop**.
1. Call `engine.generate(prompt)`.
2. Call `validator.validate(code)`.
3. If errors exist, append them to prompt and recurse (up to `max_retries`).




* **`cli.py`**:
* **Commands:**
* `airflow-net install`: Downloads the recommended GGUF model from your HF repo.
* `airflow-net serve`: Launches the HTTP server (for Cursor).
* `airflow-net mcp`: Launches the MCP server (for Claude).





### 2. `research/` (The Factory)

*Dependencies: `src`, `torch`, `unsloth`, `github`.*

* **`lib/mining.py`**:
* **Source:** Refactored from `lib/dag_miner.py`.
* **Crucial Integration:** It imports `DAGValidator` from `src`.
```python
from airflow_net.validation import DAGValidator
# ...
validator = DAGValidator()
errors = validator.validate(content) # Filter training data using the EXACT same logic as inference.

```




* **`scripts/`**:
* **`01_mine_dags.py`**: Runs the miner, saves raw JSONL.
* **`02_gen_instruct.py`**: Runs the instruction generator (using OpenAI/Claude API) to create the training pairs.
* **`03_upload_hf.py`**: Replaces `prepare_and_upload_dataset.py`.



### 3. Root Configuration

* **`pyproject.toml`**:
* Defines the project as an installable package.
* Defines the `[project.scripts]` entry point so users can just type `airflow-net`.


* **`Makefile`**:
* `install`: `pip install -e .` (Installs the library).
* `research-install`: `pip install -r research/requirements.txt` (Installs ML tools).
* `test`: `pytest tests/`.



---

## Workflow Summary

1. **For You ( The Researcher):**
* Run `make research-install`.
* Run `python research/scripts/01_mine_dags.py` (Uses `src` validator to clean data).
* Open `research/notebooks/finetune.ipynb` to train.
* Run `python research/scripts/03_upload_hf.py` to publish the GGUF.


2. **For the User (The Consumer):**
* `pip install airflow-net`
* `airflow-net install` (Downloads your GGUF).
* **Cursor User:** `airflow-net serve` -> Connect Cursor to `localhost:8000`.
* **Claude User:** Add `airflow-net mcp` to Claude Desktop config.