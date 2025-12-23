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

# Load the dataset
dataset = load_dataset("{repo_name}")

# Filter by source
airflow_only = dataset["train"].filter(lambda x: x["source"] == "airflow")
magpie_only = dataset["train"].filter(lambda x: x["source"] == "magpie")

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

## Intended Use

This dataset is designed for fine-tuning code generation models, particularly:
- **Qwen2.5-Coder** series models (1.5B, 3B, 7B, etc.)
- Models supporting ChatML format
- Specialized Airflow DAG generation while maintaining general Python skills
- Domain adaptation with knowledge retention

## Training Recommendations

### Hyperparameters
- **Model:** Qwen/Qwen2.5-Coder-1.5B-Instruct or larger
- **Method:** LoRA/QLoRA fine-tuning (recommended: Unsloth)
- **Epochs:** 3-5 epochs
- **Batch Size:** 4-8 (with gradient accumulation to effective batch size of 32)
- **Learning Rate:** 1e-4 to 2e-4
- **LoRA rank:** 16-32
- **Max Sequence Length:** 4096 tokens

### System Prompts for Inference

For **Airflow DAG generation**:
```
You are an expert Apache Airflow developer. Generate complete, valid, and executable Airflow DAG code based on the given requirements. Respond with Python code that follows Airflow best practices.
```

For **general Python tasks**:
```
You are an expert Python developer. Provide complete, working code solutions for programming tasks. Include brief explanations of key concepts when helpful.
```

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
