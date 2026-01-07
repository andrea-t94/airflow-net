# Contributing to Airflow-Net

Thank you for your interest in contributing to Airflow-Net! We welcome contributions from everyone. By participating in this project, you help us create the best local AI agent for Apache Airflow.

## Getting Started

### Prerequisites

-   **Python 3.10+**: Ensure you have Python installed.
-   **uv**: We use `uv` for blazing fast dependency management.
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### Installation

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/YOUR_USERNAME/airflow-net.git
    cd airflow-net
    ```
3.  **Set up the environment**:
    ```bash
    # Create virtualenv and install dependencies (including dev tools)
    uv pip install -e ".[dev,research]"
    ```

## Project Structure

Here is a quick overview to help you navigate:

*   `src/airflow_net/`: **Core Application**
    *   `agent.py`: Top-level logic for handling user requests.
    *   `engine.py`: Inference engine that talks to `llama.cpp`.
    *   `server_manager.py`: Manages the background model server process.
    *   `cli.py`: The `airflow-net` command-line interface.
    *   `interfaces/`: Specialized entry points (e.g., `mcp` for Claude/Cursor).
*   `research/`: **Experimental Lab**
    *   Contains the pipeline for mining data and fine-tuning the model.
*   `tests/`: **Integration Tests**
    *   `test_integration.py`: End-to-end black box tests.

## Development Workflow

1.  **Create a Branch**:
    Always create a new branch for your work. Use a descriptive name:
    ```bash
    git checkout -b feature/add-new-command
    # or
    git checkout -b fix/inference-bug
    ```

2.  **Make your changes**.

3.  **Test Your Changes**:
    Before submitting, please run the integration tests to ensure nothing is broken.
    ```bash
    # This will spin up the server and run real inference
    pytest tests/test_integration.py
    ```

4.  **Add New Tests**:
    If you are adding a new feature or fixing a bug, **please add a new test case** in `tests/test_integration.py` (or a new test file) to cover your changes.
    *   *Bug fix?* Add a test that reproduces the bug (and passes with your fix).
    *   *New feature?* Add a smoke test to ensure the command runs successfully.

## Submitting a Pull Request

1.  Push your changes to your fork:
    ```bash
    git push origin feature/add-new-command
    ```
2.  Open a **Pull Request** against the `main` branch of the original repository.
3.  **Description**: Please describe your changes clearly. If it fixes an issue, link to it (e.g., `Fixes #123`).
4.  **CI Checks**: Our GitHub Actions will automatically run the test suite on your PR. Ensure they pass!

## Release Process (Maintainers Only)

If you are a maintainer, use our automated script to release new versions:

```bash
# 1. Ensure you are on main and up to date
git checkout main && git pull

# 2. Run the release script (patch, minor, or major)
python scripts/release.py patch

# 3. Push the tags
git push origin main
git push origin --tags
```

## Need Help?

If you have questions, feel free to open a [GitHub Issue](https://github.com/andrea-t94/airflow-net/issues) for discussion.

Happy Coding! 🚀
