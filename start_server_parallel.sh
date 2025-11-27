#!/bin/bash
# Starts the C++ Server with 4 Parallel Slots

# Path to your model
MODEL_PATH="/Users/andreatamburri/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct-GGUF/snapshots/c62434db644497c0ee545c690bb66a67eba6eb3f/qwen2-1_5b-instruct-q4_k_m.gguf"

./llama.cpp/build/bin/llama-server \
    --model "$MODEL_PATH" \
    --n-gpu-layers 99 \
    --ctx-size 2048 \
    --parallel 4 \
    --cont-batching \
    --flash-attn on \
    --port 8000
