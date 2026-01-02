#!/usr/bin/env python3

import sys
import logging
import argparse
import json
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, Optional
import openai
# from datasets import load_dataset -> Moved to local scope

from .prompts import DEFAULT_SYSTEM_PROMPT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaServerDAGGenerator:
    """Generate Airflow DAGs by querying a local llama.cpp server."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", workers: int = 1, model_card: str = "unknown"):
        self.client = openai.OpenAI(base_url=base_url, api_key="sk-no-key-required")
        self.workers = workers
        self.model_card = model_card
        # Verify connection
        try:
            self.client.models.list()
            logger.info(f"SUCCESS: Connected to Llama Server at {base_url}")
        except Exception as e:
            raise ConnectionError(f"Could not connect to server at {base_url}")

    def _extract_code(self, response: str) -> str:
        """Extract Python code from the response."""
        if "```python" in response:
            start_idx = response.find("```python") + len("```python")
            end_idx = response.find("```", start_idx)
            if end_idx != -1:
                return response[start_idx:end_idx].strip()

        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                return parts[1].strip()

        return response.strip()

    def _generate_single_dag_task(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Worker task to process a single record."""
        # Extract instruction from ChatML format
        user_msg = next((m['content'] for m in record['messages'] if m['role'] == 'user'), '')
        # Extract instruction and airflow version from user message
        parts = user_msg.split('\n\nAirflow Version:')
        instruction = parts[0].strip()
        airflow_version = parts[1].strip() if len(parts) > 1 else '2.7.2'
        metadata = record.get('metadata') or {}

        try:
            # Call the Server API
            # Using optimal params from previous evaluations:
            # - max_tokens=2048 (recommended for ~90% coverage)
            # - temperature=0.1 (deterministic generation)
            response = self.client.chat.completions.create(
                model="qwen",  # Name doesn't matter for local server
                messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"{instruction}\n\nAirflow Version: {airflow_version}"}
                ],
                max_tokens=2048,
                temperature=0.1,
                top_p=0.9,
                stop=["<|im_end|>"], # Still good to have, though chat template usually handles it
            )
            
            generated_text = response.choices[0].message.content
            clean_code = self._extract_code(generated_text)

            # Convert to ChatML format for Qwen Coder 2.5 training
            return {
                'messages': [
                    {
                        'role': 'system',
                        'content': DEFAULT_SYSTEM_PROMPT
                    },
                    {
                        'role': 'user',
                        'content': f"{instruction}\n\nAirflow Version: {airflow_version}"
                    },
                    {
                        'role': 'assistant',
                        'content': clean_code
                    }
                ],
                'metadata': {
                    **metadata,
                    'model': 'qwen-server',
                    'model_card': self.model_card,
                    'backend': 'llama-server-client',
                    'airflow_version': airflow_version
                }
            }

        except Exception as e:
            logger.error(f"Error processing DAG: {e}")
            return None

    def generate(self, instruction: str, airflow_version: str = "2.7.2") -> str:
        """Generate a single DAG from an instruction string."""
        try:
            response = self.client.chat.completions.create(
                 model="qwen",
                 messages=[
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": f"{instruction}\n\nAirflow Version: {airflow_version}"}
                 ],
                 max_tokens=2048,
                 temperature=0.1,
                 top_p=0.9,
                 stop=["<|im_end|>"]
            )
            return self._extract_code(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error generating DAG: {e}")
            raise e


def main():
    parser = argparse.ArgumentParser(description="Llama Server DAG Generation Client")
    parser.add_argument('--input', default=None, help='Input JSONL file (for local files)')
    parser.add_argument('--output', help='Output JSONL file (auto-generated if not specified)')
    parser.add_argument('--test-run', action='store_true', help='Use test dataset (local file)')
    parser.add_argument('--url', default='http://localhost:8000/v1', help='Llama Server URL')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel requests')

    # HuggingFace dataset arguments
    parser.add_argument('--hf-dataset', default='andrea-t94/airflow-dag-dataset',
                       help='HuggingFace dataset name (default: andrea-t94/airflow-dag-dataset)')
    parser.add_argument('--hf-split', default='test',
                       help='Dataset split to use: train, eval, or test (default: test)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of records to process (default: all)')
    parser.add_argument('--use-local', action='store_true',
                       help='Use local JSONL file instead of HuggingFace dataset')

    # Model identification
    parser.add_argument('--model-card', default='qwen2.5-1.5b-airflow-instruct',
                       help='Model card/identifier (default: qwen2.5-1.5b-airflow-instruct). Used for output filename and metadata')

    args = parser.parse_args()

    # Determine input source
    use_hf = not args.use_local and not args.test_run

    # Handle input paths
    if args.test_run:
        # Legacy test-run mode using local files
        args.input = args.input or 'datasets/processed/test_airflow_instructions.jsonl'
        use_hf = False
    elif args.use_local:
        # Explicit local file mode
        args.input = args.input or 'datasets/processed/airflow_instructions.jsonl'

    # Generate output filename if not specified
    if not args.output:
        if use_hf:
            # Extract dataset name from HF path (e.g., 'andrea-t94/airflow-dag-dataset' -> 'airflow-dag-dataset')
            dataset_name = args.hf_dataset.split('/')[-1]
            output_filename = f"generated_dags_{dataset_name}_{args.hf_split}_{args.model_card}.jsonl"
        else:
            output_filename = f'generated_dags_server_{args.model_card}.jsonl'

        # Determine output directory
        if args.test_run:
            args.output = f'datasets/eval/{output_filename}'
        else:
            args.output = f'inference/{output_filename}'

    try:
        generator = LlamaServerDAGGenerator(base_url=args.url, workers=args.workers, model_card=args.model_card)

        if use_hf:
            stats = generator.generate_dags_from_dataset(
                input_file=None,
                output_file=args.output,
                hf_dataset=args.hf_dataset,
                hf_split=args.hf_split,
                limit=args.limit
            )
        else:
            stats = generator.generate_dags_from_dataset(
                input_file=args.input,
                output_file=args.output,
                limit=args.limit
            )

        print(f"\nSUCCESS: Generation complete!")
        print(f"INFO: Generated: {stats['total_generated']} DAGs")
        print(f"INFO: Rate: {stats['generation_rate']:.2f} DAGs/sec")
        print(f"INFO: Saved to: {stats['output_file']}")
        return 0

    except Exception as e:
        logger.error(f"ERROR: Failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())