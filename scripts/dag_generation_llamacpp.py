#!/usr/bin/env python3

import sys
import logging
import argparse
import json
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, Any
import openai

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaServerDAGGenerator:
    """Generate Airflow DAGs by querying a local llama.cpp server."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", workers: int = 1):
        self.client = openai.OpenAI(base_url=base_url, api_key="sk-no-key-required")
        self.workers = workers
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
        instruction = record.get('instruction', '')
        input_data = record.get('input', {})
        airflow_version = input_data.get('airflow_version', '2.7.2')
        metadata = record.get('metadata', {})

        prompt = self._create_prompt(instruction, airflow_version)

        try:
            # Call the Server API
            response = self.client.completions.create(
                model="qwen",  # Name doesn't matter for local server
                prompt=prompt,
                max_tokens=1024,
                temperature=0.1,
                top_p=0.9,
                stop=["<|im_end|>", "```"],
                echo=False
            )
            
            generated_text = response.choices[0].text
            clean_code = self._extract_code(generated_text)

            return {
                'instruction': instruction,
                'input': {'airflow_version': airflow_version},
                'output': clean_code,
                'metadata': {
                    **metadata,
                    'model': 'qwen-server',
                    'backend': 'llama-server-client'
                }
            }

        except Exception as e:
            logger.error(f"Error processing DAG: {e}")
            return None

    def generate_dags_from_dataset(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Generate DAGs from a JSONL dataset using parallel server requests."""
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read Records
        records = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        logger.info(f"Processing {len(records)} records using {self.workers} workers")
        logger.info(f"Input: {input_file}")
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
    parser.add_argument('--input', default='datasets/processed/airflow_instructions.jsonl')
    parser.add_argument('--output', help='Output JSONL file')
    parser.add_argument('--test-run', action='store_true', help='Use test dataset')
    parser.add_argument('--url', default='http://localhost:8000/v1', help='Llama Server URL')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel requests')

    args = parser.parse_args()

    # Handle test-run paths
    if args.test_run:
        args.input = 'datasets/processed/test_airflow_instructions.jsonl'
        if not args.output:
            args.output = 'datasets/eval/generated_dags_server.jsonl'
    elif not args.output:
        args.output = 'datasets/inference/generated_dags_server.jsonl'

    try:
        generator = LlamaServerDAGGenerator(base_url=args.url, workers=args.workers)
        stats = generator.generate_dags_from_dataset(args.input, args.output)

        print(f"\n✅ Generation complete!")
        print(f"📊 Rate: {stats['generation_rate']:.2f} DAGs/sec")
        print(f"📁 Saved to: {stats['output_file']}")
        return 0

    except Exception as e:
        logger.error(f"❌ Failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())