import click
import logging
import sys
import subprocess
import os
import time
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional
from huggingface_hub import hf_hub_download

from airflow_net.agent import AirflowAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_REPO = "andrea-t94/qwen2.5-1.5b-airflow-instruct-gguf"
DEFAULT_FILENAME = "qwen2.5-1.5b-instruct.Q4_K_M.gguf" 
CONFIG_DIR = Path.home() / ".airflow_net"
CONFIG_FILE = CONFIG_DIR / "config.json"

def _load_config() -> Dict[str, Any]:
    """Loads configuration from ~/.airflow_net/config.json."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load config: {e}")
        return {}

def _save_config(config: Dict[str, Any]):
    """Saves configuration to ~/.airflow_net/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)

def _get_server_cmd(model_path: str, host: str = "0.0.0.0", port: int = 8000, 
                   layers: int = 99, ctx: int = 34816) -> list:
    """Constructs the command to run the llama.cpp server."""
    return [
        sys.executable, "-m", "llama_cpp.server",
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
        "--n_gpu_layers", str(layers),
        "--n_ctx", str(ctx),
        "--n_batch", "2048",
        "--flash_attn", "true",
    ]

def _resolve_model_path(model_path: str = None, hf_repo: str = None, hf_file: str = None) -> str:
    """
    Resolves the model path based on arguments:
    1. If model_path is provided, checks if it exists locally.
    2. If hf_repo/hf_file are provided, downloads from HF.
    3. If neither, downloads the default model from HF.
    """
    if model_path:
        path = Path(model_path)
        if not path.exists():
            raise click.ClickException(f"ERROR: Model file not found at {model_path}")
        return str(path)

    # If no local model specified, determine which HF model to download
    repo_id = hf_repo or DEFAULT_REPO
    filename = hf_file or DEFAULT_FILENAME

    click.echo(f"INFO: Ensuring model is available from {repo_id} ({filename})...")
    try:
        # huggingface_hub handles caching automatically
        cached_path = hf_hub_download(repo_id=repo_id, filename=filename)
        click.echo(f"SUCCESS: Model available at: {cached_path}")
        return cached_path
    except Exception as e:
        raise click.ClickException(f"ERROR: Downloading model: {e}")

@click.group()
def main():
    """Airflow-Net CLI: Agentic DAG Generation."""
    pass

@main.command()
@click.option('--set-version', help="Set the default Airflow version.")
@click.option('--show', is_flag=True, help="Show current configuration.")
def config(set_version, show):
    """Manage Airflow-Net configuration."""
    cfg = _load_config()
    
    if set_version:
        cfg['airflow_version'] = set_version
        _save_config(cfg)
        click.echo(f"SUCCESS: Default Airflow version set to {set_version}")
    
    if show:
        click.echo("Current Configuration:")
        if not cfg:
            click.echo("  (Empty)")
        for k, v in cfg.items():
            click.echo(f"  {k}: {v}")
    
    if not set_version and not show:
        click.echo(click.get_current_context().get_help())

@main.command()
@click.option('--hf-repo', help=f"Hugging Face Repo ID (default: {DEFAULT_REPO})")
@click.option('--hf-file', help=f"Model filename (default: {DEFAULT_FILENAME})")
def install(hf_repo, hf_file):
    """Downloads the model (without starting server)."""
    click.echo("Installing model...")
    _resolve_model_path(model_path=None, hf_repo=hf_repo, hf_file=hf_file)
    click.echo("SUCCESS: Installation complete.")

@main.command()
@click.option('--host', default="0.0.0.0", help="Host to bind to.")
@click.option('--port', default=8000, help="Port to bind to.")
@click.option('--model', help="Path to local GGUF model file.")
@click.option('--hf-repo', help="Clean Override: Hugging Face Repo ID to download from.")
@click.option('--hf-file', help="Clean Override: Hugging Face filename to download.")
@click.option('--layers', default=99, help="Number of GPU layers (default: 99 for max GPU).")
@click.option('--ctx', default=34816, help="Context size (default: 34816).")
@click.option('--workers', default=8, help="Number of parallel workers.")
@click.option('--detach', '-d', is_flag=True, help="Run server in background (detached).")
def serve(host, port, model, hf_repo, hf_file, layers, ctx, workers, detach):
    """Launches the HTTP server (OpenAI-compatible) using llama-cpp-python."""
    
    # Resolve the model path (auto-download if needed)
    try:
        final_model_path = _resolve_model_path(model, hf_repo, hf_file)
    except Exception as e:
        click.echo(e)
        return

    click.echo(f"INFO: Starting Airflow-Net Server on {host}:{port}...")
    click.echo(f"INFO: Model: {final_model_path}")
    
    cmd = _get_server_cmd(final_model_path, host, port, layers, ctx)
    env = os.environ.copy()
    
    if detach:
        click.echo(f"INFO: Running in background (detached)...")
        # Start a new session to fully detach
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL,
            start_new_session=True, # Detach from terminal
            env=env
        )
        click.echo(f"SUCCESS: Server started in background with PID {process.pid}.")
        click.echo(f"Run 'airflow-net stop' to stop it.")
        return

    click.echo(f"Executing server (Ctrl+C to stop)...")
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        click.echo("\nStopping server...")
    except Exception as e:
        click.echo(f"Server failed: {e}")

@main.command()
def stop():
    """Stops any running background server instances."""
    try:
        # Find python processes running llama_cpp.server
        # simple pgrep might kill other things, strict pattern match is better
        # This is mac/linux specific
        click.echo("Turning off servers...")
        cmd = ["pkill", "-f", "llama_cpp.server"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        click.echo("SUCCESS: Background servers stopped.")
    except subprocess.CalledProcessError:
        click.echo("INFO: No running servers found.")
    except Exception as e:
         click.echo(f"ERROR: Could not stop server: {e}")

def _ensure_server_running(url: str):
    """Checks if server is running, and starts a background detached one if not."""
    
    # 1. Check if server is already running
    try:
        requests.get(f"{url}/models", timeout=1)
        click.echo(f"INFO: Connected to {url}")
        return
    except (requests.RequestException, Exception):
        pass # Server not up

    # 2. Logic to start server
    if "localhost" not in url and "127.0.0.1" not in url:
        raise click.ClickException(f"ERROR: Could not connect to remote server {url}")
        
    click.echo("INFO: Server not running. Starting background server (will remain running)...")
    
    model_path = _resolve_model_path()
    
    # Parse port
    port = 8000
    if ":" in url.split("/")[-2]:
        try:
            port = int(url.split(":")[-1].split("/")[0])
        except:
            pass
            
    cmd = _get_server_cmd(model_path, port=port)
    
    # Start detached
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.DEVNULL,
        start_new_session=True 
    )
    
    # Wait for ready
    click.echo(f"INFO: Waiting for model to load found at PID {process.pid}...")
    
    # Poll for 90 seconds
    for _ in range(90):
        try:
            requests.get(f"{url}/models", timeout=1)
            click.echo("SUCCESS: Server ready. You can now run 'airflow-net chat' instantly.")
            click.echo("IMPORTANT: Run 'airflow-net stop' when you are done to free up resources.")
            return
        except:
            time.sleep(1)
            
    raise click.ClickException("ERROR: Timed out waiting for server to start.")

@main.command()
@click.option('--instruction', '-i', required=True, help="Instruction for DAG generation.")
@click.option('--airflow-version', help="Target Airflow version (overrides config).")
@click.option('--output', '-o', help="Optional output file path.")
@click.option('--url', default="http://localhost:8000/v1", help="Server URL.")
def chat(instruction, airflow_version, output, url):
    """Generates a DAG. Auto-starts a persistent background server if needed."""
    
    try:
        # Resolve Airflow Version
        if airflow_version:
             target_version = airflow_version
        else:
            cfg = _load_config()
            target_version = cfg.get('airflow_version')
            
            if not target_version:
                 target_version = click.prompt("First time setup: Enter default Airflow version", default="2.7.2")
                 cfg['airflow_version'] = target_version
                 _save_config(cfg)
                 click.echo(f"INFO: Saved default version {target_version} to configuration.")

        click.echo(f"INFO: Using Airflow Version: {target_version}")

        # Ensure server is up
        _ensure_server_running(url)
        
        # Now connect agent
        agent = AirflowAgent(server_url=url)

        click.echo(f"INFO: Instruction: {instruction}")
        click.echo("INFO: Generating...")

        result = agent.generate_dag(instruction, airflow_version=target_version)
        
        if result["success"]:
            click.echo("\nSUCCESS: Generation Successful!\n")
            click.echo(result["code"])
            
            if output:
                with open(output, "w") as f:
                    f.write(result["code"])
                click.echo(f"\nINFO: Saved to {output}")
        else:
            click.echo("\nERROR: Generation Failed.")
            if "errors" in result:
                click.echo("Validation Errors:")
                for err in result["errors"]:
                    click.echo(f"- {err}")

    except Exception as e:
        click.echo(f"ERROR: {e}")

@main.command()
def mcp():
    """Launches the MCP server for Claude."""
    click.echo("Starting MCP server...")
    # TODO: Import and run the MCP server from interfaces.mcp
    click.echo("Not yet implemented: MCP Server start")

if __name__ == '__main__':
    main()
