#!/usr/bin/env python3



import logging
from pathlib import Path
from typing import Dict, Any, Optional
import openai


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



