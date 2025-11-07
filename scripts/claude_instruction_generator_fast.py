#!/usr/bin/env python3
"""
High-Performance Claude Instruction Generator with Concurrent Processing
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FastClaudeInstructionGenerator:
    """High-performance version with concurrent processing."""

    def __init__(self, api_key: str, max_workers: int = 10):
        self.api_key = api_key
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        self.max_workers = max_workers

    def _create_analysis_prompt(self, dag_record: Dict) -> str:
        """Create comprehensive analysis prompt for Claude to generate TWO instructions."""
        content = dag_record['content']
        metadata = dag_record['metadata']

        # Truncate content if too long
        if len(content) > 4000:
            content = content[:4000] + "\n# ... (code truncated)"

        prompt = f"""You are an expert Airflow instructor. Analyze this Airflow DAG code and create TWO different learning instructions.

AIRFLOW DAG CODE:
```python
{content}
```

BASIC METADATA:
- File: {metadata.get('file_name', 'unknown')}
- Airflow Version: {metadata.get('airflow_version', 'unknown')}
- Line Count: {metadata.get('line_count', 0)}
- Operators: {', '.join(metadata.get('operators', [])[:5])}

Please analyze this DAG and provide your response in the following JSON format:

{{
  "primary_instruction": {{
    "instruction": "A clear, educational instruction focusing on the main learning objective",
    "complexity_score": 1-3,
    "complexity_reasoning": "Brief explanation of why you assigned this complexity",
  }},
  "alternative_instruction": {{
    "instruction": "A different perspective or focus on the same DAG code",
    "complexity_score": 1-3,
    "complexity_reasoning": "Brief explanation of why you assigned this complexity",
  }}
}}

GUIDELINES:
1. CREATE TWO DIFFERENT INSTRUCTIONS:
   - PRIMARY: Focus on the main functionality and core learning objective
   - ALTERNATIVE: Different angle (e.g., operational focus, specific feature, architectural pattern)

2. BOTH INSTRUCTIONS should be clear, actionable, and explain WHAT to build, not HOW.
   - Keep instructions concise (max 2-3 sentences each)
   - Stay within our 600 token response limit

3. COMPLEXITY SCORING (1-3 scale for BOTH):

   **SCORE 1 (Easy):**
   - ≤ 2 different operators (BashOperator, PythonOperator, EmptyOperator)
   - < 50 lines of code
   - Simple sequential dependencies (A >> B)
   - Only standard Airflow imports
   - No external integrations

   **SCORE 2 (Medium):**
   - 3-5 different operators OR external service integrations
   - 50-150 lines of code
   - Parallel tasks, branching, or sensors
   - Cloud service operators (AWS, GCP, Azure)
   - Database connections
   - Some custom logic or functions

   **SCORE 3 (Hard):**
   - 6+ operators OR advanced Airflow features
   - > 150 lines OR complex custom code
   - TaskFlow API, dynamic task mapping, custom operators
   - Multiple external services orchestration
   - Complex dependencies, error handling, custom hooks
   - Advanced patterns (task groups, XCom, complex branching)

4. COMPLEXITY REASONING: Explain SPECIFICALLY why you assigned this score based on the criteria above in maximum 100 tokens.

Respond with ONLY the JSON object, no additional text."""

        return prompt

    def _call_claude_api(self, prompt: str, max_retries: int = 3) -> tuple[Optional[Dict], str]:
        """Call Claude API and parse JSON response. Returns (result, error_reason)."""

        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        # Setup session with connection pooling and retries
        session = requests.Session()

        # Retry strategy for connection issues
        retry_strategy = Retry(
            total=2,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        session.mount("https://", adapter)

        for attempt in range(max_retries):
            try:
                payload = {
                    "model": "claude-3-5-haiku-20241022",
                    "max_tokens": 600,  
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": prompt}]
                }

                response = session.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=45  # Increased timeout for better reliability
                )

                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        content = response_data['content'][0]['text'].strip()

                        # Clean up JSON formatting
                        if content.startswith('```json'):
                            content = content.replace('```json', '').replace('```', '').strip()

                        analysis = json.loads(content)

                        # Validate structure
                        if 'primary_instruction' in analysis and 'alternative_instruction' in analysis:
                            for key in ['primary_instruction', 'alternative_instruction']:
                                instruction = analysis[key]
                                required_fields = ['instruction', 'complexity_score', 'complexity_reasoning']
                                if not all(field in instruction for field in required_fields):
                                    return None, 'json_parsing_errors'

                                score = instruction.get('complexity_score', 0)
                                if score == 1:
                                    instruction['complexity_level'] = 'easy'
                                elif score == 2:
                                    instruction['complexity_level'] = 'medium'
                                elif score == 3:
                                    instruction['complexity_level'] = 'hard'
                                else:
                                    return None, 'json_parsing_errors'

                            return analysis, 'success'
                        else:
                            return None, 'json_parsing_errors'

                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON parsing failed: {e}")
                        return None, 'json_parsing_errors'
                    except KeyError as e:
                        logger.debug(f"Missing response field: {e}")
                        return None, 'json_parsing_errors'

                elif response.status_code == 429:
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                elif response.status_code == 413:
                    # Payload too large - token limit exceeded
                    return None, 'token_limit_exceeded'

                else:
                    logger.error(f"API call failed: {response.status_code}")
                    return None, 'api_timeouts'

            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None, 'api_timeouts'

            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None, 'network_errors'

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue

        # If we get here, all retries failed
        return None, 'rate_limits' if response.status_code == 429 else 'other_errors'

    def generate_instruction(self, dag_record: Dict) -> tuple[List[Dict], str]:
        """Generate TWO instructions for a single DAG. Returns (instructions, error_reason)."""

        if not dag_record['metadata'].get('syntax_valid', True):
            return [], 'invalid_syntax_skipped'

        prompt = self._create_analysis_prompt(dag_record)
        analysis, error_reason = self._call_claude_api(prompt)

        if not analysis:
            return [], error_reason

        instructions = []

        # Extract primary instruction
        if 'primary_instruction' in analysis:
            primary = analysis['primary_instruction']
            primary['instruction_type'] = 'primary'
            instructions.append(primary)

        # Extract alternative instruction
        if 'alternative_instruction' in analysis:
            alternative = analysis['alternative_instruction']
            alternative['instruction_type'] = 'alternative'
            instructions.append(alternative)

        return instructions, 'success'

    def process_dataset(self, input_file: str, output_file: str, max_dags: Optional[int] = None, progress_callback=None) -> Dict:
        """Process dataset with thread pool for synchronous calls."""

        instructions = []
        stats = {
            'total_processed': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'complexity_distribution': {'easy': 0, 'medium': 0, 'hard': 0},
            'api_calls': 0,
            'estimated_cost': 0.0,
            'failure_reasons': {
                'json_parsing_errors': 0,
                'api_timeouts': 0,
                'rate_limits': 0,
                'network_errors': 0,
                'token_limit_exceeded': 0,
                'invalid_syntax_skipped': 0,
                'other_errors': 0
            }
        }

        # Load all DAG records
        dag_records = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_dags and len(dag_records) >= max_dags:
                    break

                try:
                    dag_record = json.loads(line.strip())
                    if dag_record['metadata'].get('syntax_valid', True):
                        dag_records.append(dag_record)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Processing {len(dag_records)} DAGs with {self.max_workers} threads...")

        # Process with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_dag = {
                executor.submit(self.generate_instruction, dag_record): dag_record
                for dag_record in dag_records
            }

            completed = 0
            # Process completed tasks
            for future in as_completed(future_to_dag):
                dag_record = future_to_dag[future]

                try:
                    generated_instructions, error_reason = future.result()

                    if generated_instructions:
                        for analysis in generated_instructions:
                            instruction_record = {
                                'instruction': analysis['instruction'],
                                'input': {
                                    'airflow_version': dag_record['metadata']['airflow_version'],
                                    'operators': dag_record['metadata']['operators'],
                                    'line_count': dag_record['metadata']['line_count'],
                                    'is_multifile': dag_record['metadata']['is_multifile']
                                },
                                'output': dag_record['content'],
                                'metadata': {
                                    'file_name': dag_record['metadata']['file_name'],
                                    'instruction_source': 'claude-3.5',
                                    'instruction_type': analysis['instruction_type'],
                                    'claude_complexity_score': analysis['complexity_score'],
                                    'claude_complexity_level': analysis['complexity_level'],
                                    'claude_complexity_reasoning': analysis['complexity_reasoning'],
                                    'key_concepts': analysis.get('key_concepts', []),
                                    'learning_objectives': analysis.get('learning_objectives', [])
                                }
                            }

                            instructions.append(instruction_record)
                            stats['successful_generations'] += 1
                            stats['complexity_distribution'][analysis['complexity_level']] += 1

                        stats['api_calls'] += 1
                    else:
                        stats['failed_generations'] += 1
                        # Track specific failure reason
                        if error_reason in stats['failure_reasons']:
                            stats['failure_reasons'][error_reason] += 1
                        else:
                            stats['failure_reasons']['other_errors'] += 1

                except Exception as e:
                    logger.error(f"Error processing DAG {dag_record['metadata'].get('file_name', 'unknown')}: {e}")
                    stats['failed_generations'] += 1

                completed += 1
                stats['total_processed'] = completed

                if progress_callback:
                    progress_callback(completed, len(dag_records))

        # Save instructions
        with open(output_file, 'w', encoding='utf-8') as f:
            for instruction in instructions:
                json.dump(instruction, f, ensure_ascii=False)
                f.write('\n')

        # Save stats
        stats_file = output_file.replace('.jsonl', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        return stats



def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip().strip('"\'')
    return os.environ.get('ANTHROPIC_API_KEY')


def main():
    """Main function for high-performance processing."""
    import argparse

    parser = argparse.ArgumentParser(description="High-Performance Claude Instruction Generator")
    parser.add_argument('--input', required=True, help='Input JSONL file with DAG records')
    parser.add_argument('--output', required=True, help='Output JSONL file for instructions')
    parser.add_argument('--max-dags', type=int, help='Maximum number of DAGs to process')
    parser.add_argument('--workers', type=int, default=10, help='Number of concurrent workers')

    args = parser.parse_args()

    # Get API key
    api_key = load_env()
    if not api_key:
        print("❌ Please provide Claude API key via ANTHROPIC_API_KEY environment variable")
        return 1

    if not Path(args.input).exists():
        print(f"❌ Input file {args.input} not found!")
        return 1

    print(f"🚀 High-Performance Claude Instruction Generator")
    print(f"👥 Workers: {args.workers}")

    generator = FastClaudeInstructionGenerator(
        api_key=api_key,
        max_workers=args.workers
    )

    def progress_callback(completed, total):
        percentage = (completed / total) * 100
        print(f"\r📊 Progress: {completed}/{total} ({percentage:.1f}%) completed", end='', flush=True)

    start_time = time.time()

    try:
        # Use threaded processing
        stats = generator.process_dataset(
            args.input, args.output, args.max_dags, progress_callback
        )

        print()  # New line after progress

        elapsed_time = time.time() - start_time

        print(f"\n✅ Generated {stats['successful_generations']} instructions")
        print(f"📊 Complexity distribution: {stats['complexity_distribution']}")
        print(f"📈 Success rate: {stats['successful_generations']}/{stats['total_processed']} ({100*stats['successful_generations']/stats['total_processed']:.1f}%)")
        print(f"⏱️  Total time: {elapsed_time:.1f} seconds")
        print(f"⚡ Speed: {stats['total_processed']/elapsed_time:.1f} DAGs/second")

    except KeyboardInterrupt:
        print(f"\n⏹️  Generation interrupted by user")
        return 1

    return 0


if __name__ == "__main__":
    main()