# Airflow DAG Evaluation Methodology

This document outlines the comprehensive evaluation framework used to validate and score the generated Airflow DAGs. The evaluation strategy rests on two pillars: **Structural Validation** (Parser-based) and **Semantic Evaluation** (LLM-based).

---

## 1. Structural Validation (Quantitative)

We use a custom AST-based parser (`DAGValidator`) to strictly enforce that generated code is valid Python and a valid Airflow structure. This provides our quantitative "pass/fail" metrics.

### Validation Checks

These errors render the DAG unusable or unparseable.
*   **SYNTAX_ERROR**: Malformed Python code that fails `ast.parse()` (e.g., missing colons, unbalanced parens).
*   **CIRCULAR_DEPENDENCY**: A loop in the task dependency graph (e.g., `A >> B >> A`), which prevents the DAG from loading.
*   **EMPTY_DAG**: A file containing no DAG definition or task operators.
*   **PARSE_ERROR**: General runtime exceptions during AST traversal.
*   **DUPLICATE_TASK_ID**: Multiple tasks sharing the same `task_id` (causes runtime overwrites).

### Metrics
We track the **Validity Rate (%)**: The percentage of generated samples that pass all critical checks.

---

## 2. LLM Evaluation

Structural validity doesn't guarantee the code does what the user asked. For this, we use an **LLM-as-a-Judge** approach (specifically **Claude 4.5 Sonnet** via Batch API) to score the quality of the valid DAGs.

### Scoring Criteria (0/1 Scale)
The model functions as an expert Airflow Reviewer, grading on three binary criteria:

#### 1. Idiomatic Airflow
*   **Pass (1)**: Uses specific Providers and Operators designed for the task.
    *   *Example:* `from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator`
*   **Fail (0)**: Relies on generic "Pythonic" patterns where it wraps logic in a `PythonOperator` + Hook instead of using the native Operator.

#### 2. No Hallucination/Leakage
*   **Pass (1)**: Code is clean, production-ready, and uses only standard Airflow libraries.
*   **Fail (0)**: Code exhibits any of the following:
    *   Imports internal testing modules or test harness boilerplate (e.g., `from tests_common.test_utils.system_tests import get_test_run`).
    *   Hallucinates non-existent modules, operators, or API methods (e.g., `GoogleCampaignManagerBatchInsertOperator`).
    *   Uses parameters that do not exist for the specified operator.

#### 3. Instruction Adherence
*   **Pass (1)**: Fulfills the specific business logic requested (e.g., "load data AND validate").
*   **Fail (0)**: Misses a key step of the instruction.

---

## 3. How to Run Evaluations

Evaluations are centralized in the research notebook pipeline.

**Location**: [`research/finetuning/notebooks/03_model_evaluation.ipynb`](../research/finetuning/notebooks/03_model_evaluation.ipynb)

### Workflow
1.  **Input**: The notebook reads JSONL inference results (containing generated code) from `research/artifacts/finetuning/01_inference_results`.
2.  **Parser Step**: Runs `DAGValidator` across all samples to filter out invalid code and calculate error rates.
3.  **LLM Step**: Sends a sample of valid DAGs to the Anthropic Batch API for qualitative scoring.
4.  **Output**: Generates CSV reports and visualization plots in `research/artifacts/finetuning/02_evaluation_results`.