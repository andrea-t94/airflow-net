import sys
import subprocess
import os
import time
import requests
import logging
from pathlib import Path
from urllib.parse import urlparse
from huggingface_hub import hf_hub_download

# Configure logging to stderr to avoid interfering with MCP stdio
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(handler)

# Constants
DEFAULT_REPO = "andrea-t94/qwen2.5-1.5b-airflow-instruct-gguf"
DEFAULT_FILENAME = "qwen2.5-1.5b-instruct.Q4_K_M.gguf"

def get_server_cmd(model_path: str, host: str = "0.0.0.0", port: int = 8000, 
                   layers: int = 99, ctx: int = 4096, flash_attn: bool = False) -> list:
    """Constructs the command to run the llama.cpp server."""
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
        "--n_gpu_layers", str(layers),
        "--n_ctx", str(ctx),
        "--n_batch", "512",
    ]
    if flash_attn:
        cmd.extend(["--flash_attn", "true"])
    return cmd

def resolve_model_path(model_path: str = None, hf_repo: str = None, hf_file: str = None) -> str:
    """
    Resolves the model path:
    1. If model_path is provided, checks existence.
    2. If hf_repo/hf_file provided, downloads from HF.
    3. Defaults to internal repo/filename.
    """
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return str(path)

    repo_id = hf_repo or DEFAULT_REPO
    filename = hf_file or DEFAULT_FILENAME

    logger.info(f"Ensuring model is available from {repo_id} ({filename})...")
    try:
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        logger.info(f"Model available at: {cached_path}")
        return cached_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")

def ensure_server_running(url: str):
    """Checks if server is running, and starts a background detached one if not."""
    
    # Parse URL first
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or "localhost"
        port = parsed.port or 8000
    except Exception as e:
        logger.warning(f"Failed to parse URL '{url}', defaulting to localhost:8000. Error: {e}")
        hostname = "localhost"
        port = 8000

    # 1. Check if server is already running
    try:
        requests.get(f"{url}/models", timeout=1)
        logger.info(f"Connected to {url}")
        return
    except (requests.RequestException, Exception):
        pass # Server not up

    # 2. Logic to start server
    # We only auto-start if the user is pointing to a local instance.
    # subprocess.Popen can ONLY start a process on the current machine (localhost).
    # If the user configured a remote URL (e.g. 192.168.x.x) and it's down,
    # we cannot "magically" start it remotely, so we must raise an error.
    if hostname not in ["localhost", "127.0.0.1", "0.0.0.0"]:
        raise RuntimeError(f"Could not connect to remote server {url} (and cannot auto-start remote instances)")
        
    logger.info("Server not running. Starting background server (will remain running)...")
    
    model_path = resolve_model_path()
            
    # Auto-start with defaults
    cmd = get_server_cmd(model_path, port=port, flash_attn=True)
    
    # Start detached
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        start_new_session=True 
    )
    
    logger.info(f"Waiting for model to load found at PID {process.pid}...")
    
    # Poll for 90 seconds
    for _ in range(90):
        try:
            requests.get(f"{url}/models", timeout=1)
            logger.info("Server ready.")
            return
        except:
            time.sleep(1)
            
    raise TimeoutError("Timed out waiting for server to start.")
