#!/usr/bin/env python3
import time
import sys
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Error: transformers not installed.")
    sys.exit(1)

# Configuration
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
P99_PROMPT_LEN = 74
P99_GEN_LEN = 64 # Short generation for quick comparison, or full for comprehensive? Let's do a shorter sample first to test viability.
                  # Actually user wants "benchmarking against plain". We should try to match P99 load if possible, 
                  # but Transformers might be painfully slow on CPU. Let's aim for 128 gen tokens to get a t/s reading.

def run_benchmark():
    print(f"Loading model: {MODEL_ID} (Transformers)")
    
    device = "cpu"
    print(f"Device: {device}")

    # Attempt to load with bitsandbytes (4-bit) if requested
    # Note: BnB generally requires CUDA. On Mac/MPS it usually fails or falls back.
    # We will try standard loading first for "Plain Inference" baseline.
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        print("Loading model (Standard FP32/FP16)...")
        # Trying float16 for MPS optimization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32, # CPU usually prefers float32
        ).to(device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\nStarting Benchmarks (Transformers)...")
    
    # Input
    input_text = "def qsort(arr): " * 10 # Roughly 60-70 tokens?
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_tokens = inputs.input_ids.shape[1]
    print(f"Input Tokens: {input_tokens}")

    # Generate
    print("Generating...")
    start_time = time.perf_counter()
    
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128, 
        do_sample=True,
        temperature=0.1
    )
    
    end_time = time.perf_counter()
    
    gen_tokens = outputs.shape[1] - input_tokens
    duration = end_time - start_time
    speed = gen_tokens / duration
    
    print(f"\nResults:")
    print(f"Generated: {gen_tokens} tokens")
    print(f"Duration:  {duration:.2f} s")
    print(f"Speed:     {speed:.2f} t/s")

if __name__ == "__main__":
    run_benchmark()
