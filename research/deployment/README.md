# Deployment Benchmarks

This directory contains benchmarking scripts to evaluate the performance of different inference backends for the Airflow-Net model.

## Overview

We benchmark two approaches:
1. **llama.cpp (Metal)**: Our production stack using `llama-cpp-python` with Metal GPU acceleration
2. **Transformers (CPU)**: Standard Hugging Face Transformers baseline

## Results Summary

| Backend | Hardware | Generation Speed | Notes |
|:---|:---|:---|:---|
| **llama.cpp** | M1 Pro (Metal) | **~65 t/s** | Production stack (Q4_K_M quantization) |
| **Transformers** | M1 Pro (CPU) | **~4 t/s** | Baseline (FP32, CPU fallback) |

**Key Finding**: Our `llama.cpp` + Metal stack is **>15x faster** than standard CPU inference.

### Context Size Impact

We tested with varying context sizes (P90: 1908, P95: 2344, P99: 2935, PROD: 4096 tokens):

- **Generation speed remained constant** (~64-68 t/s) regardless of context size
- **Total time scales linearly** with output length, not context size
- **Conclusion**: For workloads under 4k tokens, context allocation is essentially "free" on Apple Silicon

### Deployment Recommendation

**Use the full 4096 context in production.** Since context size has negligible performance impact, there's no benefit to limiting output length (e.g., capping at P90). The system handles P99 workloads just as efficiently as P90 - the only difference is the time spent actually generating more tokens, which is unavoidable.

## Running the Benchmarks

## Running the Benchmarks

> **Note**: Ensure you have installed the project dependencies. See the [root README](../../README.md) for setup instructions.

### llama.cpp Benchmark (Production Stack)

```bash
# From project root
python research/deployment/benchmark_llama.py

# With custom model
python research/deployment/benchmark_llama.py --model /path/to/model.gguf

# CPU only (no GPU)
python research/deployment/benchmark_llama.py --layers 0
```

**Output**: Tests P90, P95, P99, and PROD scenarios with variable context sizes.

### Transformers Benchmark (Baseline Comparison)

```bash
# From project root (requires research dependencies)
uv run --with torch,transformers,accelerate python research/deployment/benchmark_transformers.py
```

**Note**: This benchmark runs on CPU because `bitsandbytes` (the standard quantization library for Transformers) doesn't support Metal. Running unquantized FP16/FP32 on MPS would require significantly more memory and hit PyTorch MPS tensor size limitations. The CPU baseline serves as a fair comparison to show the advantage of `llama.cpp`'s native Metal quantization support.

## Understanding the Results

### Metrics Explained

- **Prompt Eval (t/s)**: Speed of processing the input prompt (pre-fill phase)
- **Generation (t/s)**: Speed of generating output tokens (decode phase)
- **Total Time**: End-to-end request latency

### Scenarios

Based on token distribution analysis from our dataset:

- **P90**: 64 prompt + 1828 output tokens (~28s total)
- **P95**: 67 prompt + 2261 output tokens (~35s total)
- **P99**: 74 prompt + 2845 output tokens (~45s total)
- **PROD**: 74 prompt + 2845 output tokens with 4096 context (~44s total)

### Why llama.cpp is Faster

1. **Custom Metal Kernels**: Hand-written GPU shaders optimized for Apple Silicon
2. **4-bit Quantization**: Reduces memory bandwidth requirements by 4x
3. **Efficient KV Cache**: Optimized attention mechanism for autoregressive generation
4. **No PyTorch Overhead**: Direct hardware access without general-purpose framework layers

## Technical Details

### llama.cpp Configuration

- **Model**: Qwen2.5-Coder-1.5B-Instruct (Q4_K_M GGUF)
- **Backend**: Metal (GPU)
- **Batch Size**: 512
- **GPU Layers**: 99 (full offload)
- **Context**: Variable (1908-4096 tokens)

### Transformers Configuration

- **Model**: Qwen/Qwen2.5-Coder-1.5B-Instruct
- **Backend**: CPU (MPS fails due to PyTorch limitations)
- **Precision**: FP32
- **Generation**: 128 tokens (limited sample for comparison)

## Interpreting for Your Hardware

If you're running on different hardware:

- **M1/M2/M3 Mac**: Expect similar results to those shown
- **Intel Mac**: llama.cpp will fall back to CPU, expect ~10-15 t/s
- **Linux/Windows with NVIDIA GPU**: llama.cpp with CUDA should achieve 80-120 t/s depending on GPU
- **CPU-only**: Expect 5-15 t/s depending on CPU cores and model

## Next Steps

For production deployment optimization, see:
- `docs/03_inference_benchmarks.md` - Detailed analysis
- `../README.md` - Research directory overview
