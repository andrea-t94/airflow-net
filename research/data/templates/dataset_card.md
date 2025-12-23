---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- airflow
- code-generation
- python
- chatml
- qwen
size_categories:
- 1K<n<10K
---

# Airflow DAG Generation Dataset

This dataset combines Airflow-specific DAG generation examples with general Python coding instructions for fine-tuning code generation models.

## Dataset Description

**Total Samples:** {total_samples:,}

### Dataset Composition

The dataset includes two types of examples, identified by the `source` field:

1. **Airflow Instructions** (`source: "airflow"`) - {airflow_count:,} samples ({airflow_pct:.1f}%)
   - High-quality DAG generation examples with instruction variants
   - Domain-specific Airflow DAG code
   - Multiple instruction formulations per example
   - Covers various Airflow operators and patterns
   - System prompt: *"You are an expert Apache Airflow developer..."*

2. **Magpie General Python** (`source: "magpie"`) - {magpie_count:,} samples ({magpie_pct:.1f}%)
   - Distilled from Qwen2.5-Coder-32B using Magpie technique
   - General Python programming tasks (algorithms, data structures, libraries)
   - Includes concise explanations with working code
   - System prompt: *"You are an expert Python developer..."*
   - Prevents catastrophic forgetting of general Python capabilities

### Dataset Splits

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | {train_count:,} | 90% |
| Eval  | {eval_count:,} | 5% |
| Test  | {test_count:,} | 5% |

## Format

The dataset uses **ChatML format** compatible with Qwen2.5-Coder and other modern LLMs.

### Schema

Each example contains:
- `messages` (list): Conversation in ChatML format with roles: system, user, assistant
- `source` (string): Dataset origin - either `"airflow"` or `"magpie"`
- `metadata` (dict, optional): Additional metadata from original datasets

### Example Structure

```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "You are an expert Apache Airflow developer. Generate complete, valid, and executable Airflow DAG code based on the given requirements. Respond with Python code that follows Airflow best practices."
    }},
    {{
      "role": "user",
      "content": "Create a DAG that runs a bash script every morning at 6am."
    }},
    {{
      "role": "assistant",
      "content": "from airflow import DAG\\nfrom airflow.operators.bash import BashOperator\\nimport pendulum\\n\\nwith DAG(\\n    dag_id='morning_bash_script',\\n    schedule='0 6 * * *',\\n    start_date=pendulum.datetime(2024, 1, 1, tz='UTC'),\\n    catchup=False,\\n) as dag:\\n    run_script = BashOperator(\\n        task_id='run_morning_script',\\n        bash_command='/path/to/script.sh'\\n    )"
    }}
  ],
  "source": "airflow",
  "metadata": {{...}}
}}
```

### Using the Dataset

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load the dataset
dataset = load_dataset("{repo_name}")

# Filter by source
airflow_only = dataset["train"].filter(lambda x: x["source"] == "airflow")
magpie_only = dataset["train"].filter(lambda x: x["source"] == "magpie")

# Initialize tokenizer (example)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct")

# Apply chat template for training
def format_for_training(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {{"text": text}}

formatted_dataset = dataset.map(format_for_training)
```

### Data Fields

*   **messages**: A list of dictionaries representing the conversation. Each dictionary contains:
    *   **role**: The role of the speaker (`system`, `user`, or `assistant`).
    *   **content**: The text content of the message.
*   **source**: The origin of the data sample (`airflow` or `magpie`).
*   **metadata**: A dictionary containing additional information:
    *   For Airflow samples: May include topic classification or validation status.
    *   For Magpie samples: Includes original metadata from the source dataset.

## Source Data

### Airflow Instructions (`source: "airflow"`)
These samples are generated using a **knowledge distillation** process.
*   **Generation Method**:
    1.  **Sourcing**: We mined diverse DAG examples directly from the [official Apache Airflow repository](https://github.com/apache/airflow) to ensure alignment with official patterns and best practices.
    2.  **Validation**: Each mined DAG underwent strict validation using AST parsing and a custom validator to ensure:
        *   Valid Python syntax
        *   Unique and valid `task_id`s
        *   Absence of cyclic dependencies
    3.  **Instruction Generation**: We employed **Claude** to generate high-quality, diverse natural language instructions corresponding to these valid DAG code blocks, effectively distilling the knowledge from the codebase into instruction-response pairs.
*   **Goal**: To teach the model domain-specific syntax and best practices for Apache Airflow using verified, high-quality examples.

### Magpie General Python (`source: "magpie"`)
These samples are subsets from the [**{magpie_dataset_id}**](https://huggingface.co/datasets/{magpie_dataset_id}) dataset.
*   **Selection**: We filtered for general Python programming tasks (algorithms, data structures) to maintain the model's general coding capabilities (preventing catastrophic forgetting).
*   **Goal**: To ensure the fine-tuned model remains a competent general-purpose Python coding assistant.

## Personal and Sensitive Information

This dataset consists entirely of:
1.  **Synthetic Airflow Code**: Generating hypothetical DAGs for common use cases.
2.  **Synthetic/Distilled Python Code**: General programming problems from the Magpie dataset.

None of the data contains real user data, private keys, personally identifiable information (PII), or confidential enterprise code.

## Dataset Statistics

- **Total conversations:** {total_samples:,}
- **Airflow examples:** {airflow_count:,} ({airflow_pct:.1f}%)
- **Magpie examples:** {magpie_count:,} ({magpie_pct:.1f}%)
- **Format:** ChatML with system/user/assistant roles
- **Average tokens per example:** ~500-1000 (varies by source)

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{airflow-dag-dataset,
  title={{Airflow DAG Generation Dataset}},
  author={{Andrea Tamburri}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{repo_name}}}
}}
```

## License

Apache 2.0
