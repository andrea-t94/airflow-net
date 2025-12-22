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
    Adds source metadata to track dataset origin.

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

    # Add source metadata to each dataset
    print("\nAdding source metadata...")
    airflow_dataset = airflow_dataset.add_column("source", ["airflow"] * len(airflow_dataset))
    magpie_dataset = magpie_dataset.add_column("source", ["magpie"] * len(magpie_dataset))
    print("✓ Source metadata added")

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


def upload_to_huggingface(dataset: DatasetDict, repo_name: str, readme_content: str, private: bool = False):
    """
    Upload dataset to Hugging Face Hub with README.

    Args:
        dataset: DatasetDict to upload
        repo_name: Full repository name (e.g., "username/dataset-name")
        readme_content: Content for README.md dataset card
        private: Whether to make the dataset private
    """
    print(f"\nUploading to Hugging Face Hub: {repo_name}")
    print(f"Privacy: {'Private' if private else 'Public'}")

    # Push to hub
    dataset.push_to_hub(
        repo_name,
        private=private,
        commit_message="Upload combined Airflow + Magpie dataset with source metadata"
    )

    print(f"✓ Dataset uploaded to: https://huggingface.co/datasets/{repo_name}")

    # Upload README.md as dataset card
    print("✓ Uploading README.md dataset card...")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
        commit_message="Add dataset card with detailed documentation"
    )

    print(f"✓ Successfully uploaded dataset with README to: https://huggingface.co/datasets/{repo_name}")


def create_dataset_card(repo_name: str, dataset: DatasetDict, airflow_count: int, magpie_count: int):
    """
    Create a README.md for the dataset card with detailed format documentation.

    Args:
        repo_name: Full repository name
        dataset: DatasetDict with statistics
        airflow_count: Number of Airflow examples
        magpie_count: Number of Magpie examples
    """
    total_samples = len(dataset['train']) + len(dataset['eval']) + len(dataset['test'])
    airflow_pct = (airflow_count / total_samples) * 100
    magpie_pct = (magpie_count / total_samples) * 100

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
| Train | {len(dataset['train']):,} | 90% |
| Eval  | {len(dataset['eval']):,} | 5% |
| Test  | {len(dataset['test']):,} | 5% |

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

    # Count sources for statistics
    airflow_count = sum(1 for x in dataset['train'] if x['source'] == 'airflow') + \
                    sum(1 for x in dataset['eval'] if x['source'] == 'airflow') + \
                    sum(1 for x in dataset['test'] if x['source'] == 'airflow')

    magpie_count = sum(1 for x in dataset['train'] if x['source'] == 'magpie') + \
                   sum(1 for x in dataset['eval'] if x['source'] == 'magpie') + \
                   sum(1 for x in dataset['test'] if x['source'] == 'magpie')

    # Create dataset card
    print("\nCreating dataset card...")
    card = create_dataset_card(repo_name, dataset, airflow_count, magpie_count)

    # Write card locally first for review
    with open("dataset_card.md", "w") as f:
        f.write(card)
    print("✓ Dataset card saved to dataset_card.md")

    # Upload to Hugging Face with README
    upload_to_huggingface(dataset, repo_name, card, args.private)

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
