#!/usr/bin/env python3

import sys
import logging
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from llama_cpp import Llama

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'lib'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




class LlamaCppQwenDAGGenerator:
    """Generate Airflow DAGs using Qwen Coder with llama.cpp for fast CPU inference."""

    def __init__(self, model_path: str = None, n_ctx: int = 4096, n_threads: int = None, batch_size: int = 1):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads or -1  # Use all available cores
        self.batch_size = batch_size
        self.llm = None
        self._load_model()

    def _load_model(self):
        """Load the Qwen Coder GGUF model."""
        if not self.model_path:
            # Default to Q4_K_M model
            model_name = "Qwen/Qwen2-1.5B-Instruct-GGUF"
            model_file = "qwen2-1_5b-instruct-q4_k_m.gguf"
            logger.info(f"No model path provided. Will attempt to download: {model_name}/{model_file}")

            try:
                # Try to load from HuggingFace Hub
                from huggingface_hub import hf_hub_download
                self.model_path = hf_hub_download(repo_id=model_name, filename=model_file)
                logger.info(f"Downloaded model to: {self.model_path}")
            except ImportError:
                logger.error("huggingface_hub not installed. Run: pip install huggingface_hub")
                logger.error("Or provide a local model path with --model-path")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                sys.exit(1)

        logger.info(f"Loading model: {self.model_path}")
        logger.info(f"Context size: {self.n_ctx}")
        logger.info(f"Using {self.n_threads} threads" if self.n_threads > 0 else "Using all available threads")

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
                seed=-1,  # Random seed
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            sys.exit(1)

    def _create_prompt(self, instruction: str, airflow_version: str = "2.7.2") -> str:
        """Create a prompt for DAG generation."""
        prompt = f"""<|im_start|>system
You are an expert Apache Airflow developer. Generate a complete, valid Airflow DAG based on the given instruction.
<|im_end|>
<|im_start|>user
Create an Airflow DAG with the following requirements:

Instruction: {instruction}
Airflow Version: {airflow_version}

Requirements:
- Generate complete, syntactically correct Python code
- Include all necessary imports
- Use proper DAG structure with context manager
- Follow Airflow best practices
- Make sure the DAG is executable

Generate only the Python code for the DAG:
<|im_end|>
<|im_start|>assistant
```python
"""
        return prompt

    def _extract_code(self, response: str) -> str:
        """Extract Python code from the response."""
        # Look for code between ```python and ```
        if "```python" in response:
            start_idx = response.find("```python") + len("```python")
            end_idx = response.find("```", start_idx)
            if end_idx != -1:
                return response[start_idx:end_idx].strip()

        # Look for code between ``` and ```
        if "```" in response:
            parts = response.split("```")
            if len(parts) >= 3:
                # Take the first code block
                return parts[1].strip()

        # If no code blocks found, return the response as-is (fallback)
        return response.strip()

    def _generate_single_dag(self, instruction: str, airflow_version: str = "2.7.2") -> str:
        """Generate a single DAG from instruction."""
        prompt = self._create_prompt(instruction, airflow_version)

        try:
            # Generate with llama.cpp
            response = self.llm(
                prompt,
                max_tokens=1024,
                temperature=0.1,  # Low temperature for more deterministic output
                top_p=0.9,
                stop=["<|im_end|>", "```"],  # Stop at end token or code block end
                echo=False
            )

            generated_text = response['choices'][0]['text']
            return self._extract_code(generated_text)

        except Exception as e:
            logger.error(f"Error generating DAG: {e}")
            return f'# Error generating DAG: {str(e)}'

    def generate_dags_from_dataset(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Generate DAGs from a JSONL dataset."""
        input_path = Path(input_file)
        output_path = Path(output_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Processing dataset: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Batch size: {self.batch_size}")

        all_results = []
        total_processed = 0
        start_time = time.time()

        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line.strip())

                    instruction = record.get('instruction', '')
                    input_data = record.get('input', {})
                    airflow_version = input_data.get('airflow_version', '2.7.2')
                    metadata = record.get('metadata', {})

                    # Generate DAG
                    generated_dag = self._generate_single_dag(instruction, airflow_version)

                    result = {
                        'instruction': instruction,
                        'input': {'airflow_version': airflow_version},
                        'output': generated_dag,
                        'metadata': {
                            **metadata,
                            'model': 'qwen2-1.5b-instruct-q4_k_m-gguf',
                            'generation_id': total_processed + 1,
                            'backend': 'llama.cpp'
                        }
                    }
                    all_results.append(result)
                    total_processed += 1

                    # Log progress
                    elapsed = time.time() - start_time
                    rate = total_processed / elapsed if elapsed > 0 else 0
                    logger.info(f"Processed {total_processed} DAGs ({rate:.2f} DAGs/sec)")

                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    continue

        # Write results to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')

        elapsed_time = time.time() - start_time

        stats = {
            'total_generated': total_processed,
            'output_file': str(output_path),
            'elapsed_time': elapsed_time,
            'generation_rate': total_processed / elapsed_time if elapsed_time > 0 else 0
        }

        logger.info(f"DAG generation completed. Generated {total_processed} DAGs in {elapsed_time:.2f}s")
        logger.info(f"Results saved to: {output_path}")

        return stats


def main():
    """Main function to run DAG generation."""
    parser = argparse.ArgumentParser(description="Llama.cpp Qwen Coder DAG Generator")
    parser.add_argument('--input',
                       default='datasets/processed/airflow_instructions.jsonl',
                       help='Input JSONL dataset file')
    parser.add_argument('--output',
                       help='Output JSONL file for generated DAGs (auto-generated if not specified)')
    parser.add_argument('--test-run', action='store_true',
                       help='Use test dataset (test_airflow_instructions.jsonl) instead of full dataset')
    parser.add_argument('--model-path',
                       help='Path to GGUF model file (if not provided, will download Q4_K_M)')
    parser.add_argument('--n-ctx', type=int, default=4096,
                       help='Context size (default: 4096)')
    parser.add_argument('--n-threads', type=int,
                       help='Number of threads (default: all available)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for generation (default: 1)')

    args = parser.parse_args()

    # Handle test-run flag
    if args.test_run:
        args.input = 'datasets/processed/test_airflow_instructions.jsonl'

    # Auto-generate output path if not specified
    if not args.output:
        input_path = Path(args.input)
        dataset_name = input_path.stem  # e.g., 'airflow_instructions' or 'test_airflow_instructions'

        # Extract model name from path or use default
        if args.model_path:
            model_name = Path(args.model_path).stem.replace('.gguf', '')
        else:
            model_name = 'qwen2-1_5b-instruct-q4_k_m'

        # Determine output directory and filename
        if 'test_' in dataset_name:
            # Test runs go to eval directory
            output_dir = 'datasets/eval'
            output_filename = f"generated_dags_{model_name}.jsonl"
        else:
            # Full runs go to inference directory
            output_dir = 'datasets/inference'
            output_filename = f"{model_name}.jsonl"

        args.output = f"{output_dir}/{output_filename}"

    try:
        print("🚀 Starting Llama.cpp Qwen Coder DAG Generation")
        print(f"📁 Input: {args.input}")
        print(f"📄 Output: {args.output}")
        print(f"🤖 Model: {args.model_path or 'qwen2-1.5b-instruct-q4_k_m.gguf (auto-download)'}")
        print(f"📦 Context size: {args.n_ctx}")
        print(f"🧵 Threads: {args.n_threads or 'all available'}")
        print(f"🧪 Test run: {'Yes' if args.test_run else 'No'}")
        print()

        # Initialize generator
        generator = LlamaCppQwenDAGGenerator(
            model_path=args.model_path,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
            batch_size=args.batch_size
        )

        # Generate DAGs
        stats = generator.generate_dags_from_dataset(args.input, args.output)

        print(f"\n✅ DAG generation completed successfully!")
        print(f"📊 Statistics:")
        print(f"   Total DAGs generated: {stats['total_generated']}")
        print(f"   Generation time: {stats['elapsed_time']:.2f}s")
        print(f"   Generation rate: {stats['generation_rate']:.2f} DAGs/sec")
        print(f"📁 Results saved to: {stats['output_file']}")

        return 0

    except Exception as e:
        logger.error(f"❌ DAG generation failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())