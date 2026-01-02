import logging
from typing import Dict, Any, List

from .engine import LlamaServerDAGGenerator as ModelEngine
from .validation import DAGValidator

logger = logging.getLogger(__name__)

class AirflowAgent:
    """
    The Self-Correction Loop:
    Orchestrates Generation -> Validation -> Retry
    """
    def __init__(self, model_card: str = "qwen2.5-1.5b-airflow-instruct", server_url: str = "http://localhost:8000/v1"):
        self.engine = ModelEngine(base_url=server_url, model_card=model_card)
        self.validator = DAGValidator()
        self.max_retries = 3

    def generate_dag(self, instruction: str, airflow_version: str = "2.7.2") -> Dict[str, Any]:
        """
        Generates a DAG based on instruction, validates it, and retries if necessary.
        """
        current_instruction = instruction
        attempts = 0
        
        while attempts <= self.max_retries:
            logger.info(f"Generating DAG (Attempt {attempts + 1}/{self.max_retries + 1})...")
            
            # 1. Generate code
            code = self.engine.generate(current_instruction, airflow_version)
            
            # 2. Validate code
            errors = self.validator.validate_content(code)
            
            if not errors:
                logger.info("SUCCESS: Code generated and validated successfully!")
                return {
                    "code": code,
                    "success": True,
                    "attempts": attempts + 1,
                    "errors": []
                }
            
            # 3. If errors, append to prompt and retry
            error_msg = "\n".join([f"- {e}" for e in errors])
            logger.warning(f"Validation found issues in attempt {attempts + 1}:")
            logger.warning(error_msg)
            
            if attempts < self.max_retries:
                logger.info("Auto-correcting flaws and retrying...")
                
                # Refinement prompt (simple concatenation for now)
                # In a real agent, we might structure this as a conversation history
                current_instruction = f"{instruction}\n\nThe previous attempt had the following errors:\n{error_msg}\n\nPlease fix these errors and regenerate the DAG."
            
            attempts += 1
            
        logger.error("ERROR: Max retries reached. Could not generate valid code.")
        return {
            "code": code,
            "success": False,
            "attempts": attempts,
            "errors": [str(e) for e in errors]
        }
