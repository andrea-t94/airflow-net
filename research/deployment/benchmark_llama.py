#!/usr/bin/env python3
import time
import sys
import argparse
import logging
from typing import List, Dict, Any

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed. Run 'uv pip install llama-cpp-python[server]'")
    sys.exit(1)

# Import airflow_net helper to resolve model path
try:
    from airflow_net.server_manager import resolve_model_path
except ImportError:
    # Fallback if run out of context
    resolve_model_path = None

# Stats from analysis
SCENARIOS = {
    "P90": {"n_prompt": 64, "n_gen": 1828},
    "P95": {"n_prompt": 67, "n_gen": 2261},
    "P99": {"n_prompt": 74, "n_gen": 2845},
    "PROD": {"n_prompt": 74, "n_gen": 2845, "n_ctx": 4096}, # P99 load, but with full PROD context allocation
}

def generate_tokens(llm: Llama, n_tokens: int) -> List[int]:
    """Generates a random sequence of tokens."""
    # Just use 'the' (token 262 in many tokenizers) repeated, or random integers in vocab range
    # To be safe and fast, let's use a dummy list.
    return [1] * n_tokens # 1 is usually BOS or similar safe token

def run_benchmark(model_path: str, n_gpu_layers: int, scenarios: Dict[str, Dict[str, int]]):
    print(f"Loading model path: {model_path}")
    print(f"Config: n_gpu_layers={n_gpu_layers}, n_batch=512, Variable n_ctx")

    print("\nStarting Benchmarks (Single Request, Batch Size=512)...")
    print(f"{'Scenario':<10} | {'Ctx':<6} | {'Prompt':<8} | {'Gen':<8} | {'PP (t/s)':<10} | {'TG (t/s)':<10} | {'Total (s)':<10}")
    print("-" * 80)

    for name, config in scenarios.items():
        n_prompt = config['n_prompt']
        n_gen = config['n_gen']
        
        # Determine Context Size
        if "n_ctx" in config:
             req_ctx = config["n_ctx"]
        else:
             # Calculate required context (plus buffer)
             req_ctx = n_prompt + n_gen + 16 
        
        # Initialize Model Per Scenario
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=req_ctx,    
                n_batch=512,      
                verbose=False     
            )
        except Exception as e:
            print(f"Failed to load model for {name}: {e}")
            continue
        
        # 1. Prompt Processing (PP)
        # Create dummy prompt tokens
        prompt_tokens = [llm.tokenize(b"a")[0]] * n_prompt
        
        start_pp = time.perf_counter()
        llm.eval(prompt_tokens)
        end_pp = time.perf_counter()
        
        pp_time = end_pp - start_pp
        pp_speed = n_prompt / pp_time if pp_time > 0 else 0
        
        # 2. Token Generation (TG)
        # Sample n_gen tokens
        start_tg = time.perf_counter()
        for _ in range(n_gen):
            token = llm.sample()
            llm.eval([token])
        end_tg = time.perf_counter()
        
        tg_time = end_tg - start_tg
        tg_speed = n_gen / tg_time if tg_time > 0 else 0
        
        total_time = pp_time + tg_time
        
        print(f"{name:<10} | {req_ctx:<6} | {n_prompt:<8} | {n_gen:<8} | {pp_speed:<10.2f} | {tg_speed:<10.2f} | {total_time:<10.2f}")
        
        # Free memory
        del llm

def main():
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp model performance")
    parser.add_argument("--model", help="Path to GGUF model")
    parser.add_argument("--layers", type=int, default=99, help="GPU layers (default: 99)")
    args = parser.parse_args()

    model_path = args.model
    
    # Resolve default model if not provided
    if not model_path:
        if resolve_model_path:
            try:
                model_path = resolve_model_path()
            except Exception as e:
                print(f"Error resolving default model: {e}")
                sys.exit(1)
        else:
            print("Error: 'airflow_net' package not found and no --model provided.")
            sys.exit(1)

    run_benchmark(model_path, args.layers, SCENARIOS)

if __name__ == "__main__":
    main()
