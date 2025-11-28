#!/bin/bash
set -e # Exit on error

echo "🚀 Starting High-Performance Llama.cpp Server Installation (CMake). It requires CMake, please install it"

# 1. Define Directories
INSTALL_DIR="llama.cpp"
BUILD_DIR="build"

# 2. Clone or Update
if [ -d "$INSTALL_DIR" ]; then
    echo "📂 Directory '$INSTALL_DIR' already exists. Updating..."
    cd $INSTALL_DIR
    git pull
else
    echo "⬇️  Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp
    cd $INSTALL_DIR
fi

# 3. Clean old build artifacts if they exist
if [ -d "$BUILD_DIR" ]; then
    echo "🧹 Cleaning old build directory..."
    rm -rf $BUILD_DIR
fi

# 4. Configure with CMake (Metal/M1 is auto-detected)
echo "⚙️  Configuring CMake..."
cmake -B $BUILD_DIR -DGGML_METAL=ON

# 5. Build the Server
echo "🔨 Compiling llama-server (this may take a minute)..."
# -j uses all cores, --target llama-server builds only what we need
cmake --build $BUILD_DIR --config Release --target llama-server -j$(sysctl -n hw.logicalcpu)

# 6. Verify and Create Startup Script
# Binary is usually located in build/bin/llama-server
BINARY_PATH="$BUILD_DIR/bin/llama-server"

if [ -f "$BINARY_PATH" ]; then
    echo ""
    echo "✅ SUCCESS! The C++ server is built."
    echo "📍 Location: $(pwd)/$BINARY_PATH"

    # Go back to parent dir to save the starter script
    cd ..

    echo "📝 Creating startup script 'start_server_parallel.sh'..."

    # Note: We point to the binary inside llama.cpp/build/bin/
    cat > start_server_parallel.sh <<EOF
#!/bin/bash
# Starts the C++ Server with 4 Parallel Slots

# Path to your model
MODEL_PATH="/Users/andreatamburri/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct-GGUF/snapshots/c62434db644497c0ee545c690bb66a67eba6eb3f/qwen2-1_5b-instruct-q4_k_m.gguf"

./$INSTALL_DIR/$BINARY_PATH \\
    --model "\$MODEL_PATH" \\
    --n-gpu-layers 99 \\
    --ctx-size 2048 \\
    --parallel 4 \\
    --cont-batching \\
    --flash-attn on\\
    --port 8000
EOF

    chmod +x start_server_parallel.sh
    echo "✅ Startup script created: start_server_parallel.sh"
    echo ""
    echo "👉 To run the parallel server, execute: ./start_server_parallel.sh"
else
    echo "❌ Compilation failed. Binary not found at $BINARY_PATH"
    exit 1
fi