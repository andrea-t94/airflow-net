"""
Claude Instruction Generator Library

Generates instruction datasets from Airflow DAGs using Claude's Message Batches API.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import anthropic
import logging

from research.lib.batch_processor import ClaudeBatchProcessor
from airflow_net.prompts import DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def ensure_output_dir() -> Path:
    output_dir = Path("research/artifacts/data/02_instruct_dags")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class ClaudeBatchInstructionGenerator(ClaudeBatchProcessor):
    """Batch processing for Instruction Generation."""

    def __init__(self, api_key: str, config: Dict[str, Any] = None):
        self.config = config or {}
        poll_interval = self.config.get('api', {}).get('poll_interval', 30)
        super().__init__(api_key, poll_interval)
        self.model_config = self.config.get('model', {})

    def _create_analysis_prompt(self, dag_record: Dict) -> str:
        """Create prompt for Claude."""
        prompt_template = self.config.get('prompt_template')
        if not prompt_template:
            raise ValueError("Prompt template must be provided in configuration")

        content = dag_record['content']
        metadata = dag_record['metadata']

        # Truncate content if too long (simple safety check)
        if len(content) > 4000:
            content = content[:4000] + "\n# ... (code truncated)"

        return prompt_template.format(
            content=content,
            airflow_version=metadata.get('airflow_version', 'unknown')
        )

    def prepare_batch_requests(self, input_file: str, max_dags: Optional[int] = None) -> List[Dict]:
        """Prepare batch requests from DAG records."""
        logger.info(f"📖 Loading DAG records from {input_file}...")

        # Rely strictly on config, no silent defaults
        model = self.model_config.get('name')
        if not model:
            raise ValueError("Model name must be provided in configuration")
            
        temperature = self.model_config.get('temperature', 0.2)
        max_tokens = self.model_config.get('max_tokens', 600)

        batch_requests = []
        processed_count = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if max_dags and processed_count >= max_dags:
                    break

                try:
                    dag_record = json.loads(line.strip())
                    if not dag_record.get('metadata', {}).get('syntax_valid', True):
                        continue

                    prompt = self._create_analysis_prompt(dag_record)
                    custom_id = f"dag_{processed_count}"

                    batch_requests.append({
                        "custom_id": custom_id,
                        "params": {
                            "model": model,
                            "max_tokens": max_tokens,
                            "temperature": temperature,
                            "messages": [{"role": "user", "content": prompt}]
                        }
                    })
                    processed_count += 1

                except json.JSONDecodeError:
                    continue

        logger.info(f"📦 Prepared {len(batch_requests)} batch requests")
        return batch_requests

    def _format_as_chatml(self, dag_record: Dict, instruction_text: str, source: str, variant: int) -> Dict:
        """Format a single instruction-DAG pair as ChatML."""
        return {
            'messages': [
                {
                    'role': 'system',
                    'content': DEFAULT_SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': f"{instruction_text}\n\nAirflow Version: {dag_record['metadata'].get('airflow_version', 'unknown')}"
                },
                {
                    'role': 'assistant',
                    'content': dag_record['content']
                }
            ],
            'metadata': {
                'file_name': f"generated_variant_{variant}",
                'instruction_source': source,
                'variant_number': variant,
                'airflow_version': dag_record['metadata'].get('airflow_version')
            }
        }

    def process_batch_results(self, results: List[Dict], input_file: str, output_file: str) -> Dict:
        """Process batch results, match with original DAGs, and save."""
        logger.info("🔄 Processing results...")
        
        instruction_source = self.config.get('generation', {}).get('instruction_source', 'claude-3.5')

        # 1. Load original DAGs to map custom_id -> DAG content
        dag_map = {}
        idx = 0
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get('metadata', {}).get('syntax_valid', True):
                        dag_map[f"dag_{idx}"] = d
                        idx += 1
                except json.JSONDecodeError:
                    continue

        instructions = []
        stats = {'total': 0, 'success': 0, 'failed': 0, 'generated': 0}

        for res in results:
            stats['total'] += 1
            cid = res.get("custom_id")
            
            # Check success
            if res.get("result", {}).get("type") == "succeeded":
                try:
                    text = res["result"]["message"]["content"][0]["text"]
                    # Clean markdown code blocks
                    text = text.replace('```json', '').replace('```', '').strip()
                    
                    # Parse lines
                    parsed_objs = []
                    for line in text.split('\n'):
                        if not line.strip(): continue
                        try:
                            obj = json.loads(line)
                            if "instruction" in obj: parsed_objs.append(obj)
                        except: continue

                    if parsed_objs and cid in dag_map:
                        dag = dag_map[cid]
                        for i, p_obj in enumerate(parsed_objs):
                            instructions.append(
                                self._format_as_chatml(dag, p_obj['instruction'], instruction_source, i+1)
                            )
                        stats['success'] += 1
                        stats['generated'] += len(parsed_objs)
                    else:
                        stats['failed'] += 1
                except Exception:
                    stats['failed'] += 1
            else:
                stats['failed'] += 1

        # Save
        logger.info(f"💾 Saving {len(instructions)} instructions to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            for curr in instructions:
                f.write(json.dumps(curr, ensure_ascii=False) + '\n')
        
        # Save stats
        stats_path = Path(output_file).parent / (Path(output_file).stem + "_stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def process_dataset_batch(self, input_file: str, max_dags: int = None, is_test: bool = False) -> Dict:
        """Main workflow method."""
        # Setup paths
        output_filename = self.config.get('generation', {}).get('output_file', 'instructions.jsonl')
        if is_test:
            output_filename = "test_" + output_filename
            
        out_path = ensure_output_dir() / output_filename
        
        # Run workflow
        reqs = self.prepare_batch_requests(input_file, max_dags)
        if not reqs:
            logger.error("No valid requests prepared.")
            return {}

        batch_id = self.submit_batch(reqs)
        self.wait_for_batch_completion(batch_id)
        results = self.download_batch_results(batch_id)
        
        return self.process_batch_results(results, input_file, str(out_path))