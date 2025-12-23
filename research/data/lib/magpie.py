import json
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Any, Optional

def fetch_magpie_data(
    dataset_name: str,
    output_path: Path,
    buffer_size: int = 1500,
    streaming: bool = True,
    allowed_keywords: Optional[List[str]] = None,
    blocked_keywords: Optional[List[str]] = None
) -> int:
    """
    Fetches and filters data from the Magpie dataset.
    
    Args:
        dataset_name: Hugging Face dataset name
        output_path: Path to save the JSONL file
        buffer_size: Number of examples to collect
        streaming: Whether to stream the dataset
        allowed_keywords: Keywords that must be present in user message
        blocked_keywords: Keywords that must NOT be present in user message
        
    Returns:
        Number of examples collected
    """
    if output_path.exists():
        print(f"File {output_path} already exists. Skipping download.")
        # Count lines
        with open(output_path, "r") as f:
            return sum(1 for _ in f)

    # Defaults
    if allowed_keywords is None:
        allowed_keywords = ["python", "sql", "bash", "shell", "script", "pip", "dataframe"]
    
    if blocked_keywords is None:
        blocked_keywords = ["java ", "c++", "cpp", "rust", "golang", "react", "html", "css", "node", "typescript"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Stream-loading from {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train", streaming=streaming)
    
    count = 0
    with open(output_path, "w") as f:
        for row in dataset:
            if count >= buffer_size:
                break

            conversations = row.get('conversations', [])
            if len(conversations) < 2:
                continue

            # Extract first turn
            user_msg = next((c['value'] for c in conversations if c['from'] == 'human'), "").lower()
            assistant_msg = next((c['value'] for c in conversations if c['from'] == 'gpt'), "").lower()

            # Filtering logic
            is_relevant = any(x in user_msg for x in allowed_keywords)
            is_pollution = any(x in user_msg for x in blocked_keywords)
            has_code_syntax = ("def " in assistant_msg) or ("import " in assistant_msg) or ("```python" in assistant_msg)

            if is_relevant and not is_pollution and has_code_syntax:
                entry = {
                    "messages": [
                        {"role": "system", "content": "You are an expert Python developer. Provide complete, working code solutions for programming tasks. Include brief explanations of key concepts when helpful for understanding."},
                        {"role": "user", "content": conversations[0]['value']},
                        {"role": "assistant", "content": conversations[1]['value']}
                    ]
                }
                f.write(json.dumps(entry) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"Collected {count} examples...")

    print(f"Saved {count} Magpie examples to {output_path}")
    return count
