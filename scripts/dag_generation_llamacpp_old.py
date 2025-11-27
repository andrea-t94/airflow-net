#!/usr/bin/env python3

import sys
import logging
import argparse
import json
import time
import concurrent.futures
import queue
from pathlib import Path
from typing import List, Dict, Any
from llama_cpp import Llama

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LlamaCppQwenDAGGenerator:
    """Generate Airflow DAGs using Qwen Coder with parallel execution on M1/M2/M3."""

    def __init__(self, 
                 model_path: str = None, 
                 n_ctx: int = 2048, 
                 n_gpu_layers: int = -1,
                 workers: int = 4):  # Default to 4 parallel workers for M1 Pro
        
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.workers = workers
        self.model_queue = queue.Queue()
        
        # Initialize the model pool
        self._initialize_model_pool()

    def _initialize_model_pool(self):
        """Load multiple instances of the model for parallel processing."""
        # 1. Resolve Model Path
        if not self.model_path:
            model_name = "Qwen/Qwen2-1.5B-Instruct-GGUF"
            model_file = "qwen2-1_5b-instruct-q4_k_m.gguf"
            try:
                from huggingface_hub import hf_hub_download
                self.model_path = hf_hub_download(repo_id=model_name, filename=model_file)
                logger.info(f"Downloaded model to: {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                sys.exit(1)

        # 2. Load Workers
        logger.info(f"🚀 Initializing {self.workers} model instances for parallel inference...")
        logger.info(f"   Model: {self.model_path}")
        logger.info(f"   Context: {self.n_ctx}, GPU Layers: {self.n_gpu_layers}")

        for i in range(self.workers):
            try:
                # Optimized Llama initialization
                llm = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    n_batch=self.n_ctx,      # OPTIMIZATION: Process whole prompt at once
                    flash_attn=True,         # OPTIMIZATION: Use Flash Attention (Metal)
                    verbose=False,
                    seed=-1,
                )
                self.model_queue.put(llm)
                logger.info(f"   ✅ Loaded worker {i+1}/{self.workers}")
            except Exception as e:
                logger.error(f"Failed to load model worker {i+1}: {e}")
                sys.exit(1)

    def _create_prompt(self, instruction: str, airflow_version: str = "2.7.2") -> str:
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

    def _process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single record using a model from the queue."""
        instruction = record.get('instruction', '')
        airflow_version = record.get('input', {}).get('airflow_version', '2.7.2')
        prompt = self._create_prompt(instruction, airflow_version)
        
        # Get a model instance from the pool (blocks until one is available)
        llm = self.model_queue.get()
        try:
            response = llm(
                prompt,
                max_tokens=1024,
                temperature=0.1,
                top_p=0.9,
                stop=["<|im_end|>", "```"],
                echo=False
            )
            generated_text = response['choices'][0]['text']
            dag_code = self._extract_code(generated_text)
            
            return {
                'instruction': instruction,
                'input': {'airflow_version': airflow_version},
                'output': dag_code,
                'metadata': {
                    **record.get('metadata', {}),
                    'model': 'qwen2-1.5b-instruct-q4_k_m-gguf',
                    'backend': 'llama.cpp-parallel'
                }
            }
        except Exception as e:
            logger.error(f"Error generating DAG: {e}")
            return None
        finally:
            # Always return the model to the queue so other threads can use it
            self.model_queue.put(llm)

    def generate_dags_from_dataset(self, input_file: str, output_file: str) -> Dict[str, Any]:
        input_path = Path(input_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Read all records
        records = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

        logger.info(f"Processing {len(records)} records with {self.workers} parallel workers...")
        
        all_results = []
        start_time = time.time()
        
        # Parallel Execution Loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            # Submit all tasks
            future_to_record = {executor.submit(self._process_record, r): r for r in records}
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_record)):
                result = future.result()
                if result:
                    all_results.append(result)
                
                # Log progress
                total_processed = i + 1
                elapsed = time.time() - start_time
                rate = total_processed / elapsed
                if total_processed % 5 == 0 or total_processed == len(records):
                    logger.info(f"Processed {total_processed}/{len(records)} ({rate:.2f} DAGs/sec)")

        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')

        elapsed_time = time.time() - start_time
        return {
            'total_generated': len(all_results),
            'output_file': str(output_path),
            'elapsed_time': elapsed_time,
            'generation_rate': len(all_results) / elapsed_time if elapsed_time > 0 else 0
        }

def main():
    parser = argparse.ArgumentParser(description="Parallel Llama.cpp DAG Generator")
    parser.add_argument('--input', default='datasets/processed/airflow_instructions.jsonl')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--test-run', action='store_true')
    parser.add_argument('--model-path', help='Path to GGUF model')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel model instances')
    parser.add_argument('--n-ctx', type=int, default=2048)
    
    args = parser.parse_args()

    # Logic to handle paths (same as before)
    if args.test_run:
        args.input = 'datasets/processed/test_airflow_instructions.jsonl'
    
    if not args.output:
        # Simple output generation logic
        filename = "generated_dags_parallel.jsonl"
        directory = "datasets/eval" if args.test_run else "datasets/inference"
        args.output = f"{directory}/{filename}"

    generator = LlamaCppQwenDAGGenerator(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        workers=args.workers
    )
    
    stats = generator.generate_dags_from_dataset(args.input, args.output)
    
    print(f"\n✅ Completed! Rate: {stats['generation_rate']:.2f} DAGs/sec")
    return 0

if __name__ == "__main__":
    exit(main())