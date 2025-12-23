"""
Configuration loader for AirflowNet Research
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os
import sys

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    # Search paths relative to this file (research/data/lib/config_loader.py)
    # We want to find research/data/config/
    
    # Base is research/data
    base_dir = Path(__file__).parent.parent
    
    search_paths = [
        base_dir / "config",
        Path("research/data/config"), # From root
        Path("config"), # Legacy check from root
    ]

    config_path = None
    for folder in search_paths:
        potential_path = folder / config_file
        if potential_path.exists():
            config_path = potential_path
            break
            
    if not config_path:
        # Fallback for running from different CWD
        raise FileNotFoundError(f"Config file {config_file} not found in search paths: {[str(p) for p in search_paths]}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_mining_config() -> Dict[str, Any]:
    """Load mining configuration."""
    return load_config("mining_config.yaml")


def load_generation_config() -> Dict[str, Any]:
    """Load generation configuration."""
    return load_config("generation_config.yaml")


def load_dataset_config() -> Dict[str, Any]:
    """Load dataset creation configuration."""
    return load_config("dataset_config.yaml")


def get_api_key() -> str:
    """Get Anthropic API key from environment or .env file."""
    return _get_env_variable('ANTHROPIC_API_KEY', required=True)


def get_github_token() -> str:
    """Get GitHub token from environment or .env file."""
    return _get_env_variable('GITHUB_TOKEN', required=False)


def _get_env_variable(var_name: str, required: bool = True) -> str:
    """Get environment variable from environment or .env file."""
    value = os.environ.get(var_name)
    if value:
        return value

    env_path = Path('.env')
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip().startswith(f'{var_name}='):
                    return line.split('=', 1)[1].strip().strip('"\'')

    if required:
        raise ValueError(f"{var_name} not found in environment or .env file")
    return None


def get_input_dataset_path(config: Dict[str, Any] = None) -> str:
    """Always use dags.jsonl as input dataset."""
    dataset_path = Path("research/artifacts/01_raw_dags/dags.jsonl")
    if not dataset_path.exists():
        # check relative usage
        if Path("../../research/artifacts/01_raw_dags/dags.jsonl").exists():
             dataset_path = Path("../../research/artifacts/01_raw_dags/dags.jsonl")
             
    if not dataset_path.exists():
        raise FileNotFoundError(f"Input dataset file not found: {dataset_path}")
    return str(dataset_path)
