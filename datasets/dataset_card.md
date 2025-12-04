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

**Total Samples:** 8,238

### Dataset Composition

1. **Airflow Instructions** (~82%): High-quality DAG generation examples with instruction variants
   - Domain-specific Airflow DAG code
   - Multiple instruction formulations per example
   - Covers various Airflow operators and patterns

2. **Magpie General Python** (~18%): Distilled general Python coding instructions
   - Sourced from Qwen2.5-Coder-32B using Magpie technique
   - General Python programming tasks
   - Enhances model's general coding capabilities

### Dataset Splits

| Split | Samples | Percentage |
|-------|---------|------------|
| Train | 7,414 | 90% |
| Eval  | 412 | 5% |
| Test  | 412 | 5% |

## Format

The dataset uses **ChatML format** with a `messages` field containing conversation turns:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert Apache Airflow developer..."
    },
    {
      "role": "user",
      "content": "Design a workflow that..."
    },
    {
      "role": "assistant",
      "content": "```python\nfrom airflow import DAG..."
    }
  ]
}
```

## Intended Use

This dataset is designed for fine-tuning code generation models, particularly:
- **Qwen2.5-Coder** series models
- Models supporting ChatML format
- Airflow DAG generation tasks
- General Python code generation

## Training Recommendations

- **Model:** Qwen/Qwen2.5-Coder-1.5B-Instruct or larger
- **Method:** LoRA fine-tuning with Unsloth
- **Epochs:** 1-3 epochs
- **Batch Size:** 2-4 (with gradient accumulation)
- **Learning Rate:** 2e-4

## Citation

If you use this dataset, please cite:

```
@misc{airflow-dag-dataset,
  title={Airflow DAG Generation Dataset},
  author={Your Name},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/andrea-t94/airflow-dag-dataset}
}
```

## License

Apache 2.0
