import json
from pathlib import Path
from datasets import load_dataset

# 1. The Magpie Teacher Dataset
DATASET_NAME = "Magpie-Align/Magpie-Qwen2.5-Coder-Pro-300K-v0.1"
OUTPUT_DIR = Path("datasets/magpie")
OUTPUT_FILE = OUTPUT_DIR / "general_python_replay_buffer.jsonl"
BUFFER_SIZE = 1500

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Stream-loading from {DATASET_NAME}...")
print("Applying filters: [ALLOW: Python/SQL] | [BLOCK: Java/C++/Web]...")

# Stream mode to avoid downloading the whole 300k file
dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

with open(OUTPUT_FILE, "w") as f:
    count = 0
    for row in dataset:
        if count >= BUFFER_SIZE:
            break

        # --- Magpie uses 'conversations' list format ---
        conversations = row.get('conversations', [])

        # We need at least one user query and one assistant response
        if len(conversations) < 2:
            continue

        # Extract first turn (User -> Assistant)
        user_msg = next((c['value'] for c in conversations if c['from'] == 'human'), "").lower()
        assistant_msg = next((c['value'] for c in conversations if c['from'] == 'gpt'), "").lower()

        # --- STRICT FILTER LOGIC ---

        # 1. Must be Python related (or SQL/Bash helper tools)
        is_relevant = any(x in user_msg for x in ["python", "sql", "bash", "shell", "script", "pip", "dataframe"])

        # 2. Block Logic (No Java, C++, Frontend)
        pollution_keywords = ["java ", "c++", "cpp", "rust", "golang", "react", "html", "css", "node", "typescript"]
        is_pollution = any(x in user_msg for x in pollution_keywords)

        # 3. Syntax Check (Must look like code)
        has_code_syntax = ("def " in assistant_msg) or ("import " in assistant_msg) or ("```python" in assistant_msg)

        if is_relevant and not is_pollution and has_code_syntax:
            # Re-format to Qwen ChatML for training
            entry = {
                "messages": [
                    {"role": "system", "content": "You are an expert Python developer. Provide complete, working code solutions for programming tasks. Include brief explanations of key concepts when helpful for understanding."},
                    {"role": "user", "content": conversations[0]['value']},     # Original case
                    {"role": "assistant", "content": conversations[1]['value']}  # Original case
                ]
            }
            f.write(json.dumps(entry) + "\n")
            count += 1
            if count % 100 == 0:
                print(f"Collected {count} examples...")

print(f"Saved {count} Magpie examples to {OUTPUT_FILE}")