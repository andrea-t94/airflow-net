"""
Claude Instruction Generator Library

Generates instruction datasets from Airflow DAGs using Claude's Message Batches API.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import anthropic
import logging

logger = logging.getLogger(__name__)


def ensure_output_dir() -> Path:
    output_dir = Path("datasets") / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class ClaudeBatchInstructionGenerator:
    """Batch processing using Claude's Message Batches API via SDK."""

    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.config = config or {}

        # Set config values or defaults
        if config and 'api' in config:
            self.poll_interval = config['api'].get('poll_interval', 30)
        else:
            self.poll_interval = 30

    def _generate_custom_id(self, counter: int) -> str:
        """Generate a simple custom ID for batch requests."""
        return f'dag_{counter}'

    def _create_analysis_prompt(self, dag_record: Dict, prompt_template: str = None) -> str:
        """Create simplified prompt for Claude to generate one instruction."""
        content = dag_record['content']
        metadata = dag_record['metadata']

        # Truncate content if too long
        if len(content) > 4000:
            content = content[:4000] + "\n# ... (code truncated)"

        if prompt_template:
            return prompt_template.format(
                content=content,
                airflow_version=metadata.get('airflow_version', 'unknown')
            )

        # Default prompt
        prompt = f"""Analyze this Airflow DAG and create a clear learning instruction.

        DAG CODE:
        ```python
        {content}
        ```

        Airflow Version: {metadata.get('airflow_version', 'unknown')}

        Respond with only a JSON object:
        {{
        "instruction": "Clear, educational instruction explaining what to build (2-3 sentences max)"
        }}

        The instruction should be actionable and explain WHAT to build, not HOW to build it."""

        return prompt

    def prepare_batch_requests(self, input_file: str, max_dags: int = None,
                             prompt_template: str = None) -> List[Dict]:
        """Prepare batch requests from DAG records."""
        logger.info(f"📖 Loading DAG records from {input_file}...")

        # Get model configuration
        model_config = self.config.get('model', {})
        model = model_config.get('name', 'claude-3-5-haiku-20241022')
        temperature = model_config.get('temperature', 0.2)
        max_tokens = model_config.get('max_tokens', 600)

        batch_requests = []
        processed_count = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if max_dags and processed_count >= max_dags:
                    break

                try:
                    dag_record = json.loads(line.strip())

                    # Skip invalid syntax
                    if not dag_record['metadata'].get('syntax_valid', True):
                        continue

                    prompt = self._create_analysis_prompt(dag_record, prompt_template)

                    # Create batch request
                    request = {
                        "custom_id": self._generate_custom_id(processed_count),
                        "params": {
                            "model": model,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "messages": [{"role": "user", "content": prompt}]
                        }
                    }

                    batch_requests.append(request)
                    processed_count += 1

                except json.JSONDecodeError:
                    continue

        logger.info(f"📦 Prepared {len(batch_requests)} batch requests")
        return batch_requests

    def submit_batch(self, requests: List[Dict]) -> str:
        """Submit batch to Claude API using SDK."""
        logger.info(f"🚀 Submitting batch with {len(requests)} requests...")

        try:
            message_batch = self.client.beta.messages.batches.create(
                requests=requests
            )

            batch_id = message_batch.id
            logger.info(f"✅ Batch submitted successfully: {batch_id}")
            return batch_id

        except Exception as e:
            raise Exception(f"Failed to submit batch: {e}")

    def wait_for_batch_completion(self, batch_id: str, poll_interval: int = None) -> object:
        """Poll batch status until completion using SDK."""
        if poll_interval is None:
            poll_interval = self.poll_interval
        logger.info(f"⏳ Waiting for batch completion (polling every {poll_interval}s)...")

        start_time = time.time()
        last_status = None

        while True:
            try:
                message_batch = self.client.beta.messages.batches.retrieve(batch_id)
                status = message_batch.processing_status

                # Log status changes
                if status != last_status:
                    elapsed = time.time() - start_time
                    logger.info(f"📊 Batch status: {status} (elapsed: {elapsed:.1f}s)")

                    if getattr(message_batch, 'request_counts', None):
                        counts = message_batch.request_counts
                        total = getattr(counts, 'total', 0)
                        completed = (
                            getattr(counts, 'succeeded', 0) +
                            getattr(counts, 'errored', 0) +
                            getattr(counts, 'canceled', 0)
                        )
                        if total > 0:
                            progress = (completed / total) * 100
                            logger.info(f"📈 Progress: {completed}/{total} ({progress:.1f}%) completed")

                    last_status = status

                if status in ["ended", "failed", "canceled", "expired"]:
                    logger.info(f"🏁 Batch completed with status: {status}")
                    return message_batch

                time.sleep(poll_interval)

            except Exception as e:
                raise Exception(f"Failed to get batch status: {e}")

    def download_batch_results(self, batch_id: str) -> List[Dict]:
        """Download and parse batch results using SDK."""
        logger.info(f"⬇️ Downloading batch results...")

        try:
            # Get batch results using SDK
            results_generator = self.client.beta.messages.batches.results(batch_id)

            # Convert generator to list
            results = []
            for result in results_generator:

                # Convert SDK result to simple dict
                result_dict = {"custom_id": result.custom_id, "result": {}}

                if result.result:
                    result_dict["result"]["type"] = getattr(result.result, 'type', None)

                    if getattr(result.result, 'message', None):
                        content = getattr(result.result.message, 'content', [])
                        text = content[0].text if content else ""
                        result_dict["result"]["message"] = {"content": [{"text": text}]}

                    if getattr(result.result, 'error', None):
                        result_dict["result"]["error"] = {"type": getattr(result.result.error, 'type', "unknown")}

                results.append(result_dict)

            logger.info(f"📥 Downloaded {len(results)} results")
            return results

        except Exception as e:
            raise Exception(f"Failed to download results: {e}")

    def process_batch_results(self, results: List[Dict], input_file: str, output_file: str,
                            instruction_source: str = "claude-3.5") -> Dict:
        """Process batch results and generate final output."""
        logger.info(f"🔄 Processing batch results...")

        # Load original DAG records and recreate custom_id mapping
        dag_records = {}
        processed_count = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    dag_record = json.loads(line.strip())

                    # Skip invalid syntax (same logic as prepare_batch_requests)
                    if not dag_record['metadata'].get('syntax_valid', True):
                        continue

                    # Recreate the same custom_id logic
                    clean_id = self._generate_custom_id(processed_count)

                    # Store with the same custom_id we used for the batch
                    dag_records[clean_id] = dag_record
                    processed_count += 1

                except json.JSONDecodeError:
                    continue

        instructions = []
        stats = {
            'total_processed': 0,
            'successful_requests': 0,  # Requests that generated at least 1 instruction
            'failed_requests': 0,      # Requests that generated 0 instructions
            'total_instructions_generated': 0,  # Total instruction records created
            'expected_instructions_per_request': 3,
            'api_calls': len(results)
        }

        for result in results:
            stats['total_processed'] += 1
            custom_id = result.get("custom_id", "unknown")

            # Check if result is successful (SDK uses "succeeded" instead of "message")
            result_type = result.get("result", {}).get("type")
            if result_type == "succeeded" or result_type == "message":
                # Success case
                try:
                    message_content = result["result"]["message"]["content"][0]["text"]

                    # Clean up JSON formatting
                    if message_content.startswith('```json'):
                        message_content = message_content.replace('```json', '').replace('```', '').strip()

                    # Parse multiple JSON objects (one per line)
                    instruction_lines = [line.strip() for line in message_content.split('\n') if line.strip()]
                    parsed_instructions = []

                    for line in instruction_lines:
                        try:
                            instruction_obj = json.loads(line)
                            if "instruction" in instruction_obj:
                                parsed_instructions.append(instruction_obj)
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue

                    # Create separate records for each instruction
                    if parsed_instructions and custom_id in dag_records:
                        dag_record = dag_records[custom_id]

                        for i, analysis in enumerate(parsed_instructions):
                            instruction_record = {
                                'instruction': analysis['instruction'],
                                'input': {
                                    'airflow_version': dag_record['metadata']['airflow_version']
                                },
                                'output': dag_record['content'],
                                'metadata': {
                                    'file_name': f"{custom_id}_variant_{i+1}",
                                    'instruction_source': instruction_source,
                                    'variant_number': i + 1
                                }
                            }

                            instructions.append(instruction_record)

                        # Count this as a successful request since we got at least 1 instruction
                        stats['successful_requests'] += 1
                        stats['total_instructions_generated'] += len(parsed_instructions)
                    else:
                        stats['failed_requests'] += 1

                except (json.JSONDecodeError, Exception):
                    stats['failed_requests'] += 1

            else:
                # Error case
                stats['failed_requests'] += 1

        # Save instructions
        logger.info(f"💾 Saving {len(instructions)} instructions to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for instruction in instructions:
                json.dump(instruction, f, ensure_ascii=False)
                f.write('\n')

        # Save stats
        stats_file = output_file.replace('.jsonl', '_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"📊 Statistics saved to {stats_file}")
        return stats

    def process_dataset_batch(self, input_file: str, output_file: str, max_dags: int = None,
                            prompt_template: str = None, instruction_source: str = "claude-3.5",
                            is_test: bool = False) -> Dict:
        """Complete batch processing workflow."""
        start_time = time.time()

        # Ensure output directory exists
        output_dir = ensure_output_dir()

        # Use consistent filename (add test prefix if test mode)
        if is_test:
            final_output_file = output_dir / f"test_{output_file}"
        else:
            final_output_file = output_dir / output_file

        # Step 1: Prepare batch requests
        batch_requests = self.prepare_batch_requests(
            input_file, max_dags, prompt_template
        )

        if not batch_requests:
            logger.error("❌ No valid DAG records found")
            return {}

        # Step 2: Submit batch
        batch_id = self.submit_batch(batch_requests)

        # Step 3: Wait for completion
        self.wait_for_batch_completion(batch_id)

        # Step 4: Download results
        results = self.download_batch_results(batch_id)

        # Step 5: Process and save results
        stats = self.process_batch_results(results, input_file, str(final_output_file), instruction_source)

        # Add timing and generation metadata
        total_time = time.time() - start_time
        model_config = self.config.get('model', {})
        stats['total_time'] = total_time
        stats['batch_id'] = batch_id
        stats['generation_metadata'] = {
            'generation_timestamp': datetime.now().isoformat(),
            'is_test_run': is_test,
            'input_file': input_file,
            'output_file': str(final_output_file),
            'model': model_config.get('name', 'claude-3-5-haiku-20241022'),
            'instruction_source': instruction_source
        }

        return stats