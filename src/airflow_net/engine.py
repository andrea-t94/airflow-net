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
from datasets import load_dataset

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
            logger.info(f"✅ Connected to Llama Server at {base_url}")
        except Exception as e:
            logger.error(f"❌ Could not connect to server at {base_url}. Is it running?")
            logger.error(f"Error: {e}")
            sys.exit(1)

    def _create_prompt(self, instruction: str, airflow_version: str = "2.7.2") -> str:
        """Create a prompt for DAG generation."""
        return f"""<|im_start|>system
You are an expert Apache Airflow developer. Generate a complete, valid Airflow DAG based on the given instruction.
<|im_end|>
<|im_start|>user
Create an Airflow DAG with the following requirements:

Instruction: {instruction}
Airflow Version: {airflow_version}

Requirements:
- Generate complete, syntactically correct Python code
- Include all necessary imports
- Follow Airflow best practices
- Make sure the DAG is executable

Generate only the Python code for the DAG:
<|im_end|>
<|im_start|>assistant
```python
"""

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

        prompt = self._create_prompt(instruction, airflow_version)

        try:
            # Call the Server API
            # Using optimal params from performance_evaluation.md:
            # - max_tokens=2048 (recommended for ~90% coverage)
            # - temperature=0.1 (deterministic generation)
            response = self.client.completions.create(
                model="qwen",  # Name doesn't matter for local server
                prompt=prompt,
                max_tokens=2048,
                temperature=0.1,
                top_p=0.9,
                stop=["<|im_end|>", "```"],
                echo=False
            )
            
            generated_text = response.choices[0].text
            clean_code = self._extract_code(generated_text)

            # Convert to ChatML format for Qwen Coder 2.5 training
            return {
                'messages': [
                    {
                        'role': 'system',
                        'content': 'You are an expert Apache Airflow developer. Generate complete, valid Airflow DAGs based on given requirements.'
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

    def generate_dags_from_dataset(self, input_file: Optional[str], output_file: str,
                                  hf_dataset: Optional[str] = None,
                                  hf_split: str = "test",
                                  limit: Optional[int] = None) -> Dict[str, Any]:
        """Generate DAGs from a JSONL dataset or HuggingFace dataset using parallel server requests."""
        output_path = Path(output_file)

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read Records - either from HF or local file
        records = []
        if hf_dataset:
            logger.info(f"Loading dataset from HuggingFace: {hf_dataset} (split: {hf_split})")
            dataset = load_dataset(hf_dataset, split=hf_split, download_mode='reuse_cache_if_exists')
            records = list(dataset)
            logger.info(f"✓ Loaded {len(records)} records from HuggingFace (using cache if available)")
        else:
            input_path = Path(input_file)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")

            with open(input_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))
            logger.info(f"✓ Loaded {len(records)} records from {input_file}")

        # Apply limit if specified
        if limit and limit < len(records):
            records = records[:limit]
            logger.info(f"Limited to {limit} records")

        logger.info(f"Processing {len(records)} records using {self.workers} workers")
        logger.info(f"Output: {output_file}")

        all_results = []
        start_time = time.time()

        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            futures = [executor.submit(self._generate_single_dag_task, record) for record in records]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                result = future.result()
                if result:
                    all_results.append(result)
                
                # Logging
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                if i % 5 == 0 or i == len(records):
                    logger.info(f"Processed {i}/{len(records)} ({rate:.2f} DAGs/sec)")

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')

        total_time = time.time() - start_time
        return {
            'total_generated': len(all_results),
            'output_file': str(output_path),
            'elapsed_time': total_time,
            'generation_rate': len(all_results) / total_time
        }

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

        print(f"\n✅ Generation complete!")
        print(f"📊 Generated: {stats['total_generated']} DAGs")
        print(f"📊 Rate: {stats['generation_rate']:.2f} DAGs/sec")
        print(f"📁 Saved to: {stats['output_file']}")
        return 0

    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())