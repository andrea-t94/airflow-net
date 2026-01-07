# AirflowNet
**The First Small Language Model (SML) Specialized for Apache Airflow**

AirflowNet is a research project dedicated to creating lightweight, specialized AI models capable of generating high-quality Apache Airflow DAGs. By fine-tuning efficient base models (Qwen 2.5 coder 1.5B) on a curated dataset, we aim to bring automation to data engineering workflows, running locally on consumer hardware.


---


## Usage

### Quick Start (Recommended)

The easiest way to use `airflow-net` is to install it as a standalone tool using `uv`. This gives you the `airflow-net` command globally.

```bash
# 1. Install tool
uv tool install "git+https://github.com/andrea-t94/airflow-net.git@v0.1.19"

# 2. Chat with the Agent (Auto-starts server)
airflow-net chat -i "Create a DAG that runs dbt build every morning at 6am"
```

The `chat` command automatically starts a background server if one isn't running. Once finished, you can stop it:

```bash
airflow-net stop
```

### Configuration
The CLI persists your preferences (like target Airflow version) in `~/.airflow_net/config.json`.

**First Run Experience**:
When you run `chat` for the first time, it will interactively ask for your target Airflow version.

**Manage Config**:
```bash
# Set default Airflow version
airflow-net config --set-version 2.9.0

# Show current config
airflow-net config --show
```

### Modes

#### 1. Server Mode (`serve`)
The server hosts the LLM and provides an OpenAI-compatible API.

```bash
# Default (Foreground)
airflow-net serve

# Background (Detached)
airflow-net serve --detach

# Stop background instances
airflow-net stop

# Force CPU Mode (Hardware Agnostic)
airflow-net serve --cpu

# Explicitly disable Flash Attention (e.g. older GPUs)
airflow-net serve --no-flash-attn
```

#### Hardware Support
- **Mac (Apple Silicon)**: Fully supported with Metal acceleration.
- **Linux (Nvidia)**: Fully supported with CUDA acceleration.
- **Windows**:
  - `serve` works (CPU/CUDA) but `stop` and `--detach` commands are currently not supported.
  - Recommended to run in foreground mode on Windows.


#### 2. Chat Mode (`chat`)
The client interacts with the running server to generate DAGs.
- **Auto-Persist:** If no server is detected, `chat` starts one in the background.
- **Version Awareness:** Uses your configured Airflow version automatically.

```bash
# Basic usage (uses defaults)
airflow-net chat -i "Create a simple hello world DAG"

# Override version for one run
airflow-net chat -i "Create a DAG..." --airflow-version 2.10.0

# Save output to file
airflow-net chat -i "Create a DAG for data ingestion" -o my_dag.py
```

#### 3. MCP Mode (Claude / Cursor)

Airflow-Net implements the [Model Context Protocol (MCP)](https://modelcontextprotocol.io), allowing you to use your local specialized model as a tool inside AI assistants like **Claude Desktop**, **Claude Code**, or **Cursor**.

##### 1. Claude Code (CLI)
Create a `.mcp.json` file in your project root:
```json
{
  "mcpServers": {
    "airflow-net": {
      "command": "airflow-net",
      "args": ["mcp"]
    }
  }
}
```
Then restart Claude Code (`claude restart` or just run `claude`). It will now have access to the `generate_airflow_dag` tool.

##### 2. Cursor
1. Go to **Settings** > **Features** > **MCP**.
2. Add a new server:
   - **Name**: `airflow-net`
   - **Type**: `stdio`
   - **Command**: `airflow-net mcp`

##### Usage
Once connected, you can simply ask your assistant:
> "Generate an Airflow DAG that fetches Bitcoin prices every hour."

The assistant will delegate the task to your local Airflow-Net model (auto-starting the inference server if needed) and return the validated code.


---

## Development

If you want to contribute or run research scripts:

### Project Structure

```bash
.
├── src/airflow_net/      # Application Source
│   ├── agent.py          # Core Agent Logic
│   ├── cli.py            # CLI Implementation
│   ├── engine.py         # Inference Engine
│   └── tests/            # Integration Tests
├── research/             # Research & Training Pipeline
│   ├── data/             # Dataset Generation Scripts
│   ├── finetuning/       # Training Notebooks
└── scripts/              # Maintenance Scripts (e.g., release.py)
```

### Development
If you want to contribute, please check out our [Contribution Guidelines](CONTRIBUTING.md).


### Architecture & Internals

Airflow-Net behaves like a local client-server application:

1.  **The Database (Model)**: We use a GGUF quantized model (Qwen 2.5 derivative) powered by `llama.cpp`.
2.  **The Server (`server_manager.py`)**: Wraps `llama-cpp-python.server`. It exposes an OpenAI-compatible API at `http://localhost:8000/v1`.
    *   **Auto-Start**: The `chat` and `mcp` commands verify if the server is running. If not, they automatically spawn a background process using `server_manager.ensure_server_running()`.
3.  **The Engine (`engine.py`)**: Connects to the server API to send generation requests. It handles prompt formatting and code extraction.

### Installation

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

**For Core dev:**
```bash
uv pip install -e .
```

**For Research (Mining/Training):**
```bash
uv pip install -e ".[research]"
```

### Research Pipeline

If you want to recreate the dataset or run the research pipeline (mining, generating instructions, fine-tuning), please refer to the [Research Documentation](research/README.md). A `Makefile` is provided in the `research/` directory to orchestrate these steps.


## Documentation

-   [**01. Research Process**](docs/01_research_process.md): Deep dive into our findings, technical details on fine-tuning, and lessons learned.
-   [**02. Evaluation Methodology**](docs/02_evaluation_methodology.md): Structural and semantic evaluation criteria.
-   [**03. Inference & Benchmarks**](docs/03_inference_benchmarks.md): Hardware setup, attempts, and final performance numbers.

---