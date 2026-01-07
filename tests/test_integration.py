import pytest
import subprocess
import time
import requests
import ast
import os
import signal
import json
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
    try:
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
    finally:
        kill_server()

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



def test_mcp_dag_generation():
    """Test that the MCP server correctly handles a DAG generation request via JSON-RPC."""
    
    # 1. Start the MCP server process
    proc = subprocess.Popen(
        ["airflow-net", "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=0 # Unbuffered for reliable IPC
    )
    
    try:
        # Helper to read a single JSON-RPC line
        def read_json_response():
            start_wait = time.time()
            while time.time() - start_wait < 10: # 10s timeout for responses
                line = proc.stdout.readline()
                if line:
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue # Ignore logging/noise lines
                if proc.poll() is not None:
                     raise RuntimeError(f"MCP server exited prematurely. Stderr: {proc.stderr.read()}")
                time.sleep(0.1)
            raise TimeoutError("Timed out waiting for JSON-RPC response")

        # 2. Handshake: Initialize
        init_req = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05", # Use a recent date-based version
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.1.0"}
            }
        }
        proc.stdin.write(json.dumps(init_req) + "\n")
        proc.stdin.flush()
        
        # Read Init Response
        init_resp = read_json_response()
        assert "result" in init_resp, f"Failed handshake: {init_resp}"
        
        # 3. Handshake: Initialized Notification
        initialized_note = {
             "jsonrpc": "2.0",
             "method": "notifications/initialized"
        }
        proc.stdin.write(json.dumps(initialized_note) + "\n")
        proc.stdin.flush()
        
        # 4. Tool Call
        # We need a long timeout because this triggers model loading + generation
        tool_req = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "generate_airflow_dag",
                "arguments": {
                    "instruction": "create a simple dag that prints hello world",
                    "airflow_version": "2.9.0"
                }
            }
        }
        proc.stdin.write(json.dumps(tool_req) + "\n")
        proc.stdin.flush()
        
        # 5. Read Tool Response
        # We might have to loop over potential progress notifications or other chatter
        # But FastMCP usually just replies.
        start_wait = time.time()
        tool_resp = None
        
        while time.time() - start_wait < MODEL_TIMEOUT:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue
                
            try:
                msg = json.loads(line)
                # Look for our ID
                if msg.get("id") == 2:
                    tool_resp = msg
                    break
            except:
                pass # Ignore non-json
                
        assert tool_resp is not None, "Did not receive tool response within timeout"
        assert "result" in tool_resp, f"Tool call failed: {tool_resp}"
        assert not tool_resp.get("error"), f"Tool returned error: {tool_resp.get('error')}"
        
        content_items = tool_resp["result"].get("content", [])
        assert len(content_items) > 0
        text_content = next((c["text"] for c in content_items if c["type"] == "text"), "")
        
        # 1. Check syntax validity
        try:
            ast.parse(text_content)
        except SyntaxError:
            pytest.fail("Generated DAG code is not valid Python syntax")

        # 2. Check content (Classic or TaskFlow)
        has_classic_dag = "from airflow import DAG" in text_content or "from airflow.models.dag import DAG" in text_content
        has_taskflow_dag = "from airflow.decorators import dag" in text_content
        
        # We also check for 'Validation Status: PASS' which implies the internal validator passed
        validation_passed = "Validation Status: PASS" in tool_resp["result"].get("content", [{}])[0].get("text", "")

        assert has_classic_dag or has_taskflow_dag or validation_passed, f"Generated code does not look like a DAG:\n{text_content}"
        
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except:
            proc.kill()
