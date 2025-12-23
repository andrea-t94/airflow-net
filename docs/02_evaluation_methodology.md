# Airflow DAG Evaluation Methodology

This document outlines the comprehensive evaluation framework used to validate and score the generated Airflow DAGs. The evaluation strategy rests on two pillars: **Structural Validation** (Parser-based) and **Semantic Evaluation** (LLM-based).

---

## 1. Structural Validation (Quantitative)

We use a custom AST-based parser (`DAGValidator`) to strictly enforce that generated code is valid Python and a valid Airflow structure. This provides our quantitative "pass/fail" metrics.

### Validation Checks

#### 🔴 Critical Failures (Invalid DAG)
These errors render the DAG unusable or unparseable.
*   **SYNTAX_ERROR**: Malformed Python code that fails `ast.parse()` (e.g., missing colons, unbalanced parens).
*   **CIRCULAR_DEPENDENCY**: A loop in the task dependency graph (e.g., `A >> B >> A`), which prevents the DAG from loading.
*   **EMPTY_DAG**: A file containing no DAG definition or task operators.
*   **PARSE_ERROR**: General runtime exceptions during AST traversal.

#### ⚠️ Warnings (Poor Quality)
*   **DUPLICATE_TASK_ID**: Multiple tasks sharing the same `task_id` (causes runtime overwrites).
*   **MULTIPLE_DAGS**: Defining more than one DAG concept in a single file (bad practice for this specific use case).

### Metrics
We track the **Validity Rate (%)**: The percentage of generated samples that pass all critical checks.

---

## 2. Semantic Evaluation (Qualitative)

Structural validity doesn't guarantee the code does what the user asked. For this, we use an **LLM-as-a-Judge** approach (specifically **Claude 3.5 Sonnet** via Batch API) to score the quality of the valid DAGs.

### Scoring Criteria (1-5 Scale)
The model functions as an expert Airflow Reviewer, grading on:

#### 🎯 Correctness
*   Does the code logic match the user's natural language request?
*   Are the cron schedules, task types, and specific parameters correct?

#### 📦 Completeness
*   Are all necessary imports present?
*   Are there missing arguments or undefined variables?

#### ✨ Best Practices
*   Does it use modern Airflow operators (e.g., `BashOperator` vs deprecated patterns)?
*   Is the code pythonic and clean?

---

## 3. How to Run Evaluations

Evaluations are centralized in the research notebook pipeline.

**Location**: [`research/finetuning/notebooks/03_evaluate_generated_dags.ipynb`](../research/finetuning/notebooks/03_evaluate_generated_dags.ipynb)

### Workflow
1.  **Input**: The notebook reads JSONL inference results (containing generated code) from `research/artifacts/finetuning/01_inference_results`.
2.  **Parser Step**: Runs `DAGValidator` across all samples to filter out invalid code and calculate error rates.
3.  **LLM Step**: Sends a sample of valid DAGs to the Anthropic Batch API for qualitative scoring.
4.  **Output**: Generates CSV reports and visualization plots in `research/artifacts/finetuning/02_evaluation_results`.