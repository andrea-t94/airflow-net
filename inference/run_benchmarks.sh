#!/bin/bash
set -e

MODEL_PATH="${1:-/Users/andreatamburri/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct-GGUF/snapshots/c62434db644497c0ee545c690bb66a67eba6eb3f/qwen2-1_5b-instruct-q4_k_m.gguf}"
PROMPT_TOKENS="${2:-128}"
OUTPUT_TOKENS="${3:-4096}"
CTX_SIZE="${4:-8192}"
BATCH_SIZE="${5:-2048}"
PARALLEL_WORKERS="${6:-1,2,4,8}"

echo "📊 Running Benchmarks"
echo "Model: $MODEL_PATH"
echo "Prompt tokens: $PROMPT_TOKENS"
echo "Output tokens: $OUTPUT_TOKENS"
echo "Context size: $CTX_SIZE"
echo "Batch size: $BATCH_SIZE"
echo "Parallel workers: $PARALLEL_WORKERS"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔹 Running llama-bench (single request performance)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

./llama.cpp/build/bin/llama-bench \
    -m "$MODEL_PATH" \
    -p "$PROMPT_TOKENS" \
    -n "$OUTPUT_TOKENS" \
    -ngl 99 \
    -r 3 \
    -o md

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔹 Running llama-batched-bench (parallel workload)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

./llama.cpp/build/bin/llama-batched-bench \
    -m "$MODEL_PATH" \
    -c "$CTX_SIZE" \
    -b "$BATCH_SIZE" \
    -ub 512 \
    -npp "$PROMPT_TOKENS" \
    -ntg "$OUTPUT_TOKENS" \
    -npl "$PARALLEL_WORKERS" \
    -ngl 99

echo ""
echo "✅ Benchmarks completed"
