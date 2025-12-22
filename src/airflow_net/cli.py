import click
import logging
import sys
import subprocess
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def main():
    """Airflow-Net CLI: Agentic DAG Generation."""
    pass

@main.command()
def install():
    """Downloads the recommended GGUF model."""
    click.echo("Installing recommended models...")
    # TODO: Implement model download logic using huggingface_hub
    click.echo("Not yet implemented: Model download")

@main.command()
@click.option('--host', default="0.0.0.0", help="Host to bind to.")
@click.option('--port', default=8000, help="Port to bind to.")
@click.option('--model', help="Path to GGUF model file.")
@click.option('--layers', default=99, help="Number of GPU layers (default: 99 for max GPU).")
@click.option('--ctx', default=34816, help="Context size (default: 34816).")
@click.option('--workers', default=8, help="Number of parallel workers (Not fully supported in Python server CLI yet).")
def serve(host, port, model, layers, ctx, workers):
    """Launches the HTTP server (OpenAI-compatible) using llama-cpp-python."""
    click.echo(f"🚀 Starting Airflow-Net Server on {host}:{port}...")

    if not model:
        # Try to find a model in the cache or current dir
        click.echo("❌ Error: --model path is required.")
        return

    model_path = Path(model)
    if not model_path.exists():
         click.echo(f"❌ Error: Model file not found at {model_path}")
         return
         
    # Construct command to run python -m llama_cpp.server
    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
        "--n_gpu_layers", str(layers),
        "--n_ctx", str(ctx),
        "--n_batch", "2048", # Default from old script
        "--flash_attn", "true",
    ]

    # Environment variables for server options
    env = os.environ.copy()
    # Setting parallel workers often handled via specific env or args depending on version
    # For llama-cpp-python, we might need configuration 
    
    click.echo(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        click.echo("\nStopping server...")
    except Exception as e:
        click.echo(f"Server failed: {e}")

@main.command()
def mcp():
    """Launches the MCP server for Claude."""
    click.echo("Starting MCP server...")
    # TODO: Import and run the MCP server from interfaces.mcp
    click.echo("Not yet implemented: MCP Server start")

if __name__ == '__main__':
    main()
