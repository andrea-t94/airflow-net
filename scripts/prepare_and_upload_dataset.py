"""
Prepare combined dataset and upload to Hugging Face Hub.

This script:
1. Loads both airflow_instructions.jsonl and general_python_replay_buffer.jsonl
2. Combines them into a single dataset
3. Splits into train/eval/test sets (90%/5%/5%)
4. Uploads to Hugging Face Hub as a public dataset

Usage:
    python scripts/prepare_and_upload_dataset.py --hf_username YOUR_USERNAME

Requirements:
    pip install datasets huggingface_hub
    hf auth login  # Run this first to authenticate
"""

import argparse
from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi


def prepare_combined_dataset(airflow_path: str, magpie_path: str) -> DatasetDict:
    """
    Load and combine both datasets, then split into train/eval/test.

    Args:
        airflow_path: Path to airflow_instructions.jsonl
        magpie_path: Path to general_python_replay_buffer.jsonl

    Returns:
        DatasetDict with train/eval/test splits
    """
    print("Loading datasets...")

    # Load both datasets
    airflow_dataset = load_dataset("json", data_files=airflow_path, split="train")
    print(f"✓ Airflow dataset: {len(airflow_dataset)} samples")

    magpie_dataset = load_dataset("json", data_files=magpie_path, split="train")
    print(f"✓ Magpie dataset: {len(magpie_dataset)} samples")

    # Combine datasets
    combined_dataset = concatenate_datasets([airflow_dataset, magpie_dataset])
    print(f"✓ Combined dataset: {len(combined_dataset)} samples")

    # Verify format - both should have 'messages' field in ChatML format
    print("\nVerifying dataset format...")
    sample = combined_dataset[0]
    if "messages" not in sample:
        raise ValueError("Dataset must have 'messages' field in ChatML format")
    print(f"✓ Dataset format verified (ChatML with 'messages' field)")

    # Perform the Split (90% Train, 5% Eval, 5% Test)
    print("\nSplitting dataset...")

    # First split: Train vs (Test + Eval)
    train_testvalid = combined_dataset.train_test_split(test_size=0.1, seed=42)

    # Second split: Test vs Eval (splitting the 10% into 5% and 5%)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)

    # Recombine into a single DatasetDict
    split_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'eval': test_valid['train'],
        'test': test_valid['test']
    })

    print(f"✓ Train: {len(split_dataset['train'])} samples ({len(split_dataset['train'])/len(combined_dataset)*100:.1f}%)")
    print(f"✓ Eval:  {len(split_dataset['eval'])} samples ({len(split_dataset['eval'])/len(combined_dataset)*100:.1f}%)")
    print(f"✓ Test:  {len(split_dataset['test'])} samples ({len(split_dataset['test'])/len(combined_dataset)*100:.1f}%)")

    return split_dataset


def upload_to_huggingface(dataset: DatasetDict, repo_name: str, private: bool = False):
    """
    Upload dataset to Hugging Face Hub.

    Args:
        dataset: DatasetDict to upload
        repo_name: Full repository name (e.g., "username/dataset-name")
        private: Whether to make the dataset private
    """
    print(f"\nUploading to Hugging Face Hub: {repo_name}")
    print(f"Privacy: {'Private' if private else 'Public'}")

    # Push to hub
    dataset.push_to_hub(
        repo_name,
        private=private,
        commit_message="Initial upload of combined Airflow + Magpie dataset"
    )

    print(f"\n✓ Successfully uploaded to: https://huggingface.co/datasets/{repo_name}")


def create_dataset_card(repo_name: str, dataset: DatasetDict):
    """
    Create a README.md for the dataset card.

    Args:
        repo_name: Full repository name
        dataset: DatasetDict with statistics
    """
    total_samples = len(dataset['train']) + len(dataset['eval']) + len(dataset['test'])

    card_content = f"""---
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
| Train | {len(dataset['train']):,} | 90% |
| Eval  | {len(dataset['eval']):,} | 5% |
| Test  | {len(dataset['test']):,} | 5% |

## Format

The dataset uses **ChatML format** with a `messages` field containing conversation turns:

```json
{{
  "messages": [
    {{
      "role": "system",
      "content": "You are an expert Apache Airflow developer..."
    }},
    {{
      "role": "user",
      "content": "Design a workflow that..."
    }},
    {{
      "role": "assistant",
      "content": "```python\\nfrom airflow import DAG..."
    }}
  ]
}}
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
@misc{{airflow-dag-dataset,
  title={{Airflow DAG Generation Dataset}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/datasets/{repo_name}}}
}}
```

## License

Apache 2.0
"""

    return card_content


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and upload combined dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--hf_username",
        type=str,
        required=True,
        help="Your Hugging Face username"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="airflow-dag-dataset",
        help="Dataset name (default: airflow-dag-dataset)"
    )
    parser.add_argument(
        "--airflow_path",
        type=str,
        default="datasets/processed/airflow_instructions.jsonl",
        help="Path to airflow_instructions.jsonl"
    )
    parser.add_argument(
        "--magpie_path",
        type=str,
        default="datasets/magpie/general_python_replay_buffer.jsonl",
        help="Path to general_python_replay_buffer.jsonl"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private (default: public)"
    )

    args = parser.parse_args()

    # Full repo name
    repo_name = f"{args.hf_username}/{args.dataset_name}"

    print("=" * 60)
    print("Preparing Combined Dataset for Hugging Face Hub")
    print("=" * 60)
    print(f"Repository: {repo_name}")
    print(f"Airflow data: {args.airflow_path}")
    print(f"Magpie data: {args.magpie_path}")
    print("=" * 60)
    print()

    # Prepare dataset
    dataset = prepare_combined_dataset(args.airflow_path, args.magpie_path)

    # Upload to Hugging Face
    upload_to_huggingface(dataset, repo_name, args.private)

    # Create and upload dataset card
    print("\nCreating dataset card...")
    card = create_dataset_card(repo_name, dataset)

    # Write card locally first
    with open("dataset_card.md", "w") as f:
        f.write(card)
    print("✓ Dataset card saved to dataset_card.md")
    print("  You can review and upload it to the dataset repo manually if needed")

    print("\n" + "=" * 60)
    print("✅ COMPLETE!")
    print("=" * 60)
    print(f"\nYour dataset is now available at:")
    print(f"https://huggingface.co/datasets/{repo_name}")
    print(f"\nTo use in your notebook:")
    print(f'  dataset = load_dataset("{repo_name}")')
    print("=" * 60)


if __name__ == "__main__":
    main()
