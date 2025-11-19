# AirflowNet Makefile
# Orchestrates the complete pipeline from DAG mining to instruction generation

# Variables
PYTHON := python
VENV := venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python

# Check if virtual environment exists
VENV_EXISTS := $(shell test -d $(VENV) && echo "yes" || echo "no")

# Default target
.PHONY: help
help:
	@echo "AirflowNet Pipeline Commands:"
	@echo ""
	@echo "Setup:"
	@echo "  setup          - Create virtual environment and install dependencies"
	@echo "  clean          - Clean up generated files"
	@echo ""
	@echo "Pipeline:"
	@echo "  mine           - Mine DAGs from Airflow repositories"
	@echo "  generate       - Generate instructions from latest mined dataset"
	@echo "  pipeline       - Run complete pipeline (mine + generate)"
	@echo ""
	@echo "Development:"
	@echo "  test-mine      - Test mining with a small subset"
	@echo "  test-generate  - Test generation with 5 DAGs"
	@echo "  lint           - Run code linting"
	@echo "  status         - Show current dataset status"

# Setup
.PHONY: setup
setup:
	@echo "Setting up virtual environment..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install anthropic requests pyyaml python-dotenv
	@echo "✅ Setup complete"

# Clean up
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -f datasets/raw/*.jsonl datasets/raw/*.json
	rm -f datasets/processed/*.jsonl datasets/processed/*.json
	rm -rf $(VENV)
	rm -rf __pycache__ lib/__pycache__ scripts/__pycache__
	@echo "✅ Cleanup complete"

# Mining
.PHONY: mine
mine: check-venv
	@echo "🚀 Mining DAGs from Airflow repositories..."
	$(PYTHON_VENV) scripts/run_dag_mining.py
	@echo "✅ Mining complete"

# Instruction generation
.PHONY: generate
generate: check-venv
	@echo "🚀 Generating instructions from main dataset..."
	$(PYTHON_VENV) scripts/run_instruction_generation.py
	@echo "✅ Generation complete"

# Full pipeline
.PHONY: pipeline
pipeline: mine generate
	@echo "✅ Complete pipeline finished"

# Testing targets
.PHONY: test-mine
test-mine: check-venv
	@echo "🧪 Testing mining with latest version only..."
	$(PYTHON_VENV) scripts/run_dag_mining.py --test
	@echo "✅ Test mining complete"

.PHONY: test-generate
test-generate: check-venv
	@echo "🧪 Testing generation with 5 DAGs..."
	$(PYTHON_VENV) scripts/run_instruction_generation.py --test
	@echo "✅ Test generation complete"

# Development
.PHONY: lint
lint:
	@echo "🔍 Running linting..."
	$(PYTHON) -m py_compile lib/*.py
	$(PYTHON) -m py_compile scripts/*.py
	@echo "✅ Linting complete"

.PHONY: status
status:
	@echo "📊 Dataset Status:"
	@echo ""
	@echo "Raw datasets:"
	@ls -la datasets/raw/ 2>/dev/null || echo "  No raw datasets found"
	@echo ""
	@echo "Processed datasets:"
	@ls -la datasets/processed/ 2>/dev/null || echo "  No processed datasets found"
	@echo ""
	@echo "File structure:"
	@echo "  Production files:"
	@echo "    raw/dags.jsonl -> processed/airflow_instructions.jsonl"
	@echo "  Test files:"
	@echo "    raw/test_dags.jsonl -> processed/test_airflow_instructions.jsonl"
	@echo "  Note: Instruction generation always uses raw/dags.jsonl as input"

# Install dependencies
.PHONY: deps
deps:
	@echo "Installing dependencies..."
	$(PIP) install --upgrade anthropic requests pyyaml python-dotenv
	@echo "✅ Dependencies updated"

# Check virtual environment
.PHONY: check-venv
check-venv:
	@if [ ! -d "$(VENV)" ]; then \
		echo "❌ Virtual environment not found. Run 'make setup' first."; \
		exit 1; \
	fi

# Check environment
.PHONY: check-env
check-env:
	@echo "🔍 Checking environment..."
	@if [ -z "$$ANTHROPIC_API_KEY" ] && [ ! -f ".env" ]; then \
		echo "❌ ANTHROPIC_API_KEY not found in environment or .env file"; \
		exit 1; \
	else \
		echo "✅ ANTHROPIC_API_KEY found"; \
	fi
	@if [ -z "$$GITHUB_TOKEN" ]; then \
		echo "⚠️ GITHUB_TOKEN not found - mining will use unauthenticated requests (60/hour limit)"; \
	else \
		echo "✅ GITHUB_TOKEN found - mining will use authenticated requests (5000/hour limit)"; \
	fi

# Make directories
datasets/raw datasets/processed:
	mkdir -p $@

# Ensure directories exist for targets that need them
mine: | datasets/raw
generate: | datasets/processed