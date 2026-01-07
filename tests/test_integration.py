import pytest
import subprocess
import time
import requests
import ast
import os
import signal
from pathlib import Path

# Constants
SERVER_URL = "http://localhost:8000/v1"
HEALTH_URL = "http://localhost:8000/v1/models"
MODEL_TIMEOUT = 60 # Loading models can be slow

def is_server_running():
    try:
        requests.get(HEALTH_URL, timeout=1)
        return True
    except:
        return False

def kill_server():
    try:
        subprocess.run(["pkill", "-f", "llama_cpp.server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2) # Give it time to die
    except:
        pass

@pytest.fixture(autouse=True)
def clean_server_state():
    """Ensure no server is running before and after each test."""
    kill_server()
    yield
    kill_server()

def test_serve_starts():
    """Test that 'airflow-net serve --detach' starts a responsive server."""
    # Start server
    cmd = ["airflow-net", "serve", "--detach", "--cpu"] # Use CPU for CI/test compatibility
    subprocess.run(cmd, check=True)
    
    # Poll for startup
    start_time = time.time()
    server_up = False
    while time.time() - start_time < MODEL_TIMEOUT:
        if is_server_running():
            server_up = True
            break
        time.sleep(2)
        
    assert server_up, "Server failed to start within timeout"

def test_chat_autostart():
    """Test that 'airflow-net chat' auto-starts the server if missing."""
    assert not is_server_running()
    
    # We run a simple chat command. 
    # It will take time to load model, then generate.
    cmd = ["airflow-net", "chat", "-i", "print hello world", "--airflow-version", "2.9.0"]
    
    # We use Popen because we want to see if the server process appears *during* execution
    # But for simplicity, we can just run it and check if it succeeds (which implies server success)
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if server is running NOW (it should persist)
    assert is_server_running(), "Chat command failed to auto-start server"
    
    # Check output for success message
    assert "SUCCESS: Generation Successful" in result.stdout

def test_chat_generation(tmp_path):
    """Test that chat produces valid python code."""
    output_file = tmp_path / "test_dag.py"
    cmd = [
        "airflow-net", 
        "chat", 
        "-i", "Create a simple DAG that prints 'hello' every day", 
        "-o", str(output_file),
        "--airflow-version", "2.9.0"
    ]
    
    subprocess.run(cmd, check=True)
    
    assert output_file.exists()
    content = output_file.read_text()
    
    # 1. Check content
    assert "from airflow import DAG" in content or "from airflow.models.dag import DAG" in content
    
    # 2. Check syntax validity
    try:
        ast.parse(content)
    except SyntaxError:
        pytest.fail("Generated DAG code is not valid Python syntax")

def test_mcp_autostart():
    """Test that MCP server setup logic (which shares code with chat) triggers backend start."""
    # We can't easily run the full MCP interactive loop, but we can verify the trigger.
    # We'll use a python script that imports the same logic to test 'ensure_server_running' 
    # or just run `airflow-net mcp` and kill it after a few seconds, then check server.
    
    proc = subprocess.Popen(
        ["airflow-net", "mcp"], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for a bit (enough for server check/start logic to trigger)
    # The MCP command waits for server health check before printing "MCP Server is up"
    start_time = time.time()
    server_started = False
    
    try:
        while time.time() - start_time < MODEL_TIMEOUT:
            if is_server_running():
                server_started = True
                break
            time.sleep(2)
    finally:
        proc.terminate()
        
    assert server_started, "MCP command failed to trigger server auto-start"
