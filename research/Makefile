# AirflowNet Research Pipeline
# Orchestrates the complete pipeline from DAG mining to instruction generation
# NOTE: All python commands are run from project root to ensure imports verify correctly

# Variables
PYTHON := python
# Detect python command (use uv run if available, else standard python)
PYTHON_CMD := $(shell command -v uv >/dev/null 2>&1 && echo "uv run" || echo "python")

# Default target
.PHONY: help
help:
	@echo "AirflowNet Research Pipeline Commands:"
	@echo ""
	@echo "Pipeline Steps:"
	@echo "  mine           - Mine DAGs from Airflow repositories (Step 1)"
	@echo "  generate       - Generate instructions from latest mined dataset (Step 2)"
	@echo "  dataset        - Create final HuggingFace dataset (Step 3)"
	@echo "  pipeline       - Run complete pipeline (mine -> generate -> dataset)"
	@echo ""
	@echo "Development:"
	@echo "  test-mine      - Test mining with a small subset"
	@echo "  test-generate  - Test generation with 5 DAGs"
	@echo "  test-pipeline  - Run complete pipeline in test mode"
	@echo "  clean          - Clean up generated artifacts"
	@echo ""
	@echo "Environment:"
	@echo "  Using command: $(PYTHON_CMD)"

# -------------------------------------------------------------------------
# Step 1: Mining
# -------------------------------------------------------------------------
.PHONY: mine
mine:
	@echo "[Step 1/3] Mining DAGs from Airflow repositories..."
	cd .. && $(PYTHON_CMD) -m research.data.scripts.01_mine_dags
	@echo "Mining complete"

.PHONY: test-mine
test-mine:
	@echo "Testing mining with minimal subset..."
	cd .. && $(PYTHON_CMD) -m research.data.scripts.01_mine_dags --test
	@echo "Test mining complete"

# -------------------------------------------------------------------------
# Step 2: Instruction Generation
# -------------------------------------------------------------------------
.PHONY: generate
generate:
	@echo "[Step 2/3] Generating instructions from mined dataset..."
	cd .. && $(PYTHON_CMD) -m research.data.scripts.02_gen_instruct
	@echo "Generation complete"

.PHONY: test-generate
test-generate:
	@echo "Testing generation with 5 DAGs..."
	cd .. && $(PYTHON_CMD) -m research.data.scripts.02_gen_instruct --test
	@echo "Test generation complete"

# -------------------------------------------------------------------------
# Step 3: Dataset Creation
# -------------------------------------------------------------------------
.PHONY: dataset
dataset:
	@echo "[Step 3/3] Creating final dataset..."
	cd .. && $(PYTHON_CMD) -m research.data.scripts.03_create_dataset --dry-run
	@echo "Dataset creation complete (dry-run)"

.PHONY: dataset-upload
dataset-upload:
	@echo "[Step 3/3] Creating and UPLOADING dataset..."
	cd .. && $(PYTHON_CMD) -m research.data.scripts.03_create_dataset

# -------------------------------------------------------------------------
# Full Pipeline
# -------------------------------------------------------------------------
.PHONY: pipeline
pipeline: mine generate dataset
	@echo "Complete pipeline finished successfully!"

.PHONY: test-pipeline
test-pipeline: test-mine test-generate dataset
	@echo "Test pipeline finished successfully!"


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
.PHONY: clean
clean:
	@echo "Cleaning up artifacts..."
	@echo "Note: This does not delete the actual data in api artifacts, manually delete if needed"
	rm -f dataset_card.md
	@echo "Cleanup complete"
