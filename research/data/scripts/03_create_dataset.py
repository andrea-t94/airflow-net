#!/usr/bin/env python3
"""
Create and upload the Airflow DAG dataset.

This script:
1. Fetches/filters Magpie data if needed.
2. Loads generated Airflow instructions.
3. Combines datasets and splits into train/eval/test.
4. Generates a dataset card.
5. Uploads to Hugging Face Hub (optional).
"""

import argparse
import sys
import os
from pathlib import Path
from datasets import load_dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi

# Modify sys.path to ensure we can import from lib
current_dir = Path(__file__).resolve().parent
research_dir = current_dir.parent
sys.path.append(str(research_dir))

from lib.config_loader import load_dataset_config
from lib import magpie

def main():
    parser = argparse.ArgumentParser(description="Create and upload Airflow dataset")
    parser.add_argument("--dry-run", action="store_true", help="Skip upload to Hugging Face")
    parser.add_argument("--hf-username", help="Hugging Face username (overrides config)")
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_dataset_config()
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Airflow Dataset Creation Pipeline")
    print("=" * 60)

    # Resolve paths
    # Assuming paths in config are relative to project root or absolute
    # If they are relative, we need to locate project root
    # Heuristic: research_dir is .../research/data. Project root is .../
    project_root = research_dir.parent.parent
    
    airflow_path = project_root / config['paths']['airflow_raw']
    magpie_output_path = project_root / config['paths']['magpie_output']
    
    # 1. Ensure Magpie Data
    print(f"\n[1/5] Checking Magpie data...")
    magpie_count = magpie.fetch_magpie_data(
        dataset_name=config['magpie']['source_dataset'],
        output_path=magpie_output_path,
        buffer_size=config['magpie']['buffer_size'],
        streaming=config['magpie']['streaming'],
        allowed_keywords=config['magpie']['filters']['allowed_keywords'],
        blocked_keywords=config['magpie']['filters']['blocked_keywords']
    )
    
    # 2. Load Datasets
    print(f"\n[2/5] Loading datasets...")
    
    if not airflow_path.exists():
        print(f"Error: Airflow data not found at {airflow_path}")
        sys.exit(1)
        
    airflow_dataset = load_dataset("json", data_files=str(airflow_path), split="train")
    print(f"✓ Airflow dataset: {len(airflow_dataset)} samples")

    magpie_dataset = load_dataset("json", data_files=str(magpie_output_path), split="train")
    print(f"✓ Magpie dataset: {len(magpie_dataset)} samples")

    # Add source metadata
    airflow_dataset = airflow_dataset.add_column("source", ["airflow"] * len(airflow_dataset))
    magpie_dataset = magpie_dataset.add_column("source", ["magpie"] * len(magpie_dataset))

    # 3. Combine and Split
    print(f"\n[3/5] Processing splits...")
    combined_dataset = concatenate_datasets([airflow_dataset, magpie_dataset])
    
    # Verify format
    if "messages" not in combined_dataset[0]:
        raise ValueError("Dataset must have 'messages' field")

    # Split
    # First split: Train vs (Test + Eval)
    test_size = config['splits']['test_size']
    eval_ratio = config['splits']['eval_ratio']
    seed = config['splits']['seed']
    
    train_testvalid = combined_dataset.train_test_split(test_size=test_size, seed=seed)
    
    # Second split: Test vs Eval
    test_valid = train_testvalid['test'].train_test_split(test_size=eval_ratio, seed=seed)
    
    final_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'eval': test_valid['train'],
        'test': test_valid['test']
    })
    
    train_count = len(final_dataset['train'])
    eval_count = len(final_dataset['eval'])
    test_count = len(final_dataset['test'])
    total_samples = len(combined_dataset)

    print(f"✓ Train: {train_count} ({train_count/total_samples*100:.1f}%)")
    print(f"✓ Eval:  {eval_count} ({eval_count/total_samples*100:.1f}%)")
    print(f"✓ Test:  {test_count} ({test_count/total_samples*100:.1f}%)")

    # 4. Generate Dataset Card
    print(f"\n[4/5] Generating dataset card...")
    
    hf_username = args.hf_username or config['hf'].get('username')
    dataset_name = config['hf']['dataset_name']
    
    if not hf_username or hf_username == "${HF_USERNAME}":
         print("Warning: HF Username not set in config or arg. Using placeholder.")
         hf_username = "USER"

    repo_name = f"{hf_username}/{dataset_name}"
    
    # Count specific sources in splits for card stats
    # Simpler: just use total counts of raw loads assuming roughly equal distribution or re-count
    # Let's count properly
    def count_source(ds, src):
        return sum(1 for x in ds if x['source'] == src)

    af_count = len(airflow_dataset) # Total airflow
    mp_count = len(magpie_dataset)  # Total magpie
    
    # Calculate percentages
    af_pct = (af_count / total_samples) * 100
    mp_pct = (mp_count / total_samples) * 100
    
    # Load template
    template_path = research_dir / "templates" / "dataset_card.md"
    with open(template_path, "r") as f:
        template = f.read()
        
    card_content = template.format(
        total_samples=total_samples,
        airflow_count=af_count,
        airflow_pct=af_pct,
        magpie_count=mp_count,
        magpie_pct=mp_pct,
        train_count=train_count,
        eval_count=eval_count,
        test_count=test_count,
        test_count=test_count,
        repo_name=repo_name,
        magpie_dataset_id=config['magpie']['source_dataset']
    )
    
    with open("dataset_card.md", "w") as f:
        f.write(card_content)
    print("✓ Saved dataset_card.md")

    # 5. Upload
    print(f"\n[5/5] Uploading...")
    if args.dry_run:
        print("Dry run enabled. Skipping upload.")
    else:
        print(f"Uploading to {repo_name}...")
        try:
            final_dataset.push_to_hub(
                repo_name,
                private=config['hf']['private']
            )
            
            api = HfApi()
            api.upload_file(
                path_or_fileobj=card_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_name,
                repo_type="dataset"
            )
            print(f"✅ Upload Complete: https://huggingface.co/datasets/{repo_name}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
