import json
from pathlib import Path

def count_tokens_simple(text):
    """Simple word-based token approximation (splits on whitespace)"""
    # Rough approximation: GPT tokenizers typically produce ~1.3 tokens per word
    # We'll use a more accurate character-based estimate
    # Average: ~4 characters per token for English text
    return max(1, len(text) // 4)

def analyze_dataset(file_path):
    """Analyze token statistics for prompts (user) and DAG files (assistant)"""
    prompt_tokens = []
    dag_tokens = []
    total_tokens = []

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                messages = data.get('messages', [])

                # Extract user (prompt) and assistant (DAG file) content
                prompt = ''
                dag_file = ''

                for message in messages:
                    if message.get('role') == 'user':
                        prompt = message.get('content', '')
                    elif message.get('role') == 'assistant':
                        dag_file = message.get('content', '')

                # Estimate tokens (rough approximation: ~4 chars per token)
                prompt_tok = count_tokens_simple(prompt)
                dag_tok = count_tokens_simple(dag_file)

                prompt_tokens.append(prompt_tok)
                dag_tokens.append(dag_tok)
                total_tokens.append(prompt_tok + dag_tok)

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    # Calculate statistics
    def stats(tokens_list, name):
        if not tokens_list:
            return

        print(f"\n{name} Statistics:")
        print(f"  Total samples: {len(tokens_list)}")
        print(f"  Mean tokens: {sum(tokens_list) / len(tokens_list):.2f}")
        print(f"  Median tokens: {sorted(tokens_list)[len(tokens_list)//2]}")
        print(f"  Min tokens: {min(tokens_list)}")
        print(f"  Max tokens: {max(tokens_list)}")
        print(f"  Total tokens: {sum(tokens_list):,}")

        # Percentiles
        sorted_tokens = sorted(tokens_list)
        p50 = sorted_tokens[int(len(sorted_tokens) * 0.50)]
        p75 = sorted_tokens[int(len(sorted_tokens) * 0.75)]
        p90 = sorted_tokens[int(len(sorted_tokens) * 0.90)]
        p95 = sorted_tokens[int(len(sorted_tokens) * 0.95)]
        p99 = sorted_tokens[int(len(sorted_tokens) * 0.99)]

        print(f"  50th percentile: {p50}")
        print(f"  75th percentile: {p75}")
        print(f"  90th percentile: {p90}")
        print(f"  95th percentile: {p95}")
        print(f"  99th percentile: {p99}")

    stats(prompt_tokens, "Prompt (User)")
    stats(dag_tokens, "DAG File (Assistant)")
    stats(total_tokens, "Total (Prompt + DAG)")

    print(f"\n{'='*60}")
    print(f"Dataset: {file_path}")
    print(f"Total samples: {len(prompt_tokens)}")
    print(f"Total tokens (all samples): {sum(total_tokens):,}")
    print(f"\nNote: Token counts are approximations (~4 chars per token)")

if __name__ == "__main__":
    dataset_path = Path("datasets/processed/airflow_instructions.jsonl")
    analyze_dataset(dataset_path)