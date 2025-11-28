#!/bin/bash
set -e

echo "🔨 Building llama-bench and llama-batched-bench..."

cd llama.cpp/build

cmake --build . --target llama-bench -j$(sysctl -n hw.logicalcpu)
cmake --build . --target llama-batched-bench -j$(sysctl -n hw.logicalcpu)

cd ../..

echo "✅ Benchmarks built successfully"
