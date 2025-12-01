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
# 1. Extract the maximum number of workers from the list "1,2,4,8"
#    This converts the comma-separated string to an array and picks the last (largest) one.
IFS=',' read -ra WORKER_ARRAY <<< "$PARALLEL_WORKERS"
NUM_WORKERS=${#WORKER_ARRAY[@]}
# Get the last element using the length
MAX_WORKERS=${WORKER_ARRAY[$NUM_WORKERS-1]}
# 2. Calculate the context needed per user (Prompt + Output)
TOKENS_PER_USER=$(($PROMPT_TOKENS + $OUTPUT_TOKENS))

# 3. Calculate the TOTAL context required (Tokens per User * Max Workers)
#    We add a small buffer (+128) just to be safe.
REQUIRED_CTX=$(( $TOKENS_PER_USER * $MAX_WORKERS + 128 ))

echo "🧮 Calculated required context: $REQUIRED_CTX (for $MAX_WORKERS workers)"

./llama.cpp/build/bin/llama-batched-bench \
    -m "$MODEL_PATH" \
    -c "$REQUIRED_CTX" \
    -b "$BATCH_SIZE" \
    -ub 512 \
    -npp "$PROMPT_TOKENS" \
    -ntg "$OUTPUT_TOKENS" \
    -npl "$PARALLEL_WORKERS" \
    -ngl 99

echo ""
echo "✅ Benchmarks completed"
