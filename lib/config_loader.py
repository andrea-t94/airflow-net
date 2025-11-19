"""
Configuration loader for AirflowNet project
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, fall back to manual parsing
    pass


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path("config") / config_file

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_mining_config() -> Dict[str, Any]:
    """Load mining configuration."""
    return load_config("mining_config.yaml")


def load_generation_config() -> Dict[str, Any]:
    """Load generation configuration."""
    return load_config("generation_config.yaml")


def get_api_key() -> str:
    """Get Anthropic API key from environment or .env file."""
    return _get_env_variable('ANTHROPIC_API_KEY', required=True)


def get_github_token() -> str:
    """Get GitHub token from environment or .env file."""
    return _get_env_variable('GITHUB_TOKEN', required=False)


def _get_env_variable(var_name: str, required: bool = True) -> str:
    """Get environment variable from environment or .env file."""
    # Check environment first (dotenv should have loaded it)
    value = os.environ.get(var_name)
    if value:
        return value

    # Try .env file as fallback (in case dotenv wasn't available)
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
    dataset_path = Path("datasets/raw/dags.jsonl")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Input dataset file not found: {dataset_path}")
    return str(dataset_path)