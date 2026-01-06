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

from airflow_net.agent import AirflowAgent
from airflow_net.server_manager import (
    resolve_model_path, 
    get_server_cmd, 
    ensure_server_running,
    DEFAULT_REPO,
    DEFAULT_FILENAME
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
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
    try:
        resolve_model_path(model_path=None, hf_repo=hf_repo, hf_file=hf_file)
        click.echo("SUCCESS: Installation complete.")
    except Exception as e:
        raise click.ClickException(str(e))

@main.command()
@click.option('--host', default="0.0.0.0", help="Host to bind to.")
@click.option('--port', default=8000, help="Port to bind to.")
@click.option('--model', help="Path to local GGUF model file.")
@click.option('--hf-repo', help="Clean Override: Hugging Face Repo ID to download from.")
@click.option('--hf-file', help="Clean Override: Hugging Face filename to download.")
@click.option('--layers', default=99, help="Number of GPU layers (default: 99 for max GPU).")
@click.option('--ctx', default=4096, help="Context size (default: 4096).")
@click.option('--detach', '-d', is_flag=True, help="Run server in background (detached).")
@click.option('--cpu', is_flag=True, help="Force CPU mode (sets layers=0, disables flash attention).")
@click.option('--flash-attn/--no-flash-attn', default=None, help="Enable/Disable Flash Attention (default: auto).")
def serve(host, port, model, hf_repo, hf_file, layers, ctx, detach, cpu, flash_attn):
    """Launches the HTTP server (OpenAI-compatible) using llama-cpp-python."""
    
    # Resolve the model path (auto-download if needed)
    try:
        final_model_path = resolve_model_path(model, hf_repo, hf_file)
    except Exception as e:
        click.echo(e)
        return

    # Handle Hardware Flags
    if cpu:
        layers = 0
        if flash_attn is True:
            click.echo("WARNING: --cpu flag passed but --flash-attn requested. Ignoring flash attention.")
        flash_attn = False
    else:
        # If not CPU, default to Flash Attention unless explicitly disabled
        if flash_attn is None:
            flash_attn = True

    click.echo(f"INFO: Starting Airflow-Net Server on {host}:{port}...")
    click.echo(f"INFO: Model: {final_model_path}")
    click.echo(f"INFO: Hardware: Layers={layers}, FlashAttn={flash_attn}")
    
    cmd = get_server_cmd(final_model_path, host, port, layers, ctx, flash_attn)
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
        try:
            ensure_server_running(url)
        except Exception as e:
            raise click.ClickException(str(e))
        
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
@click.option('--url', default="http://localhost:8000/v1", help="Server URL for model backend.")
def mcp(url):
    """Launches the MCP server for Claude."""
    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    
    try:
        # Set config for the tool to pick up
        os.environ["AIRFLOW_NET_SERVER_URL"] = url
        
        from airflow_net.interfaces.mcp import mcp
        
        # Pre-warm the server so it's ready for the first request
        try:
            ensure_server_running(url)
        except Exception as e:
             logger.warning(f"Failed to pre-warm server: {e}")
        
        # Provide a clear signal that we are ready
        # Using stderr via logger is safe for MCP stdio transport
        logger.info(f"MCP Server is up and running (Backend: {url})")
             
        mcp.run()
    except ImportError:
        click.echo("ERROR: mcp library not found or interfaces.mcp not implemented.")
    except Exception as e:
        click.echo(f"ERROR: {e}")

if __name__ == '__main__':
    main()
