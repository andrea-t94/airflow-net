#!/bin/bash
set -e # Exit on error

# ========================================
# USAGE
# ========================================
# This script builds llama.cpp server binary using CMake.
# After building, use start_server_parallel.sh to run the server.
#
# Basic usage:
#   ./build_llamaserver.sh
#
# ========================================

echo "🚀 Building llama.cpp server (requires CMake)"

# Define Directories
INSTALL_DIR="llama.cpp"
BUILD_DIR="build"

# Clone or Update
if [ -d "$INSTALL_DIR" ]; then
    echo "📂 Directory '$INSTALL_DIR' already exists. Updating..."
    cd $INSTALL_DIR
    git pull
else
    echo "⬇️  Cloning llama.cpp repository..."
    git clone https://github.com/ggerganov/llama.cpp
    cd $INSTALL_DIR
fi

# Clean old build artifacts if they exist
if [ -d "$BUILD_DIR" ]; then
    echo "🧹 Cleaning old build directory..."
    rm -rf $BUILD_DIR
fi

# Configure with CMake (Metal/M1 is auto-detected)
echo "⚙️  Configuring CMake..."
cmake -B $BUILD_DIR -DGGML_METAL=ON

# Build the Server
echo "🔨 Compiling llama-server (this may take a minute)..."
# -j uses all cores, --target llama-server builds only what we need
cmake --build $BUILD_DIR --config Release --target llama-server -j$(sysctl -n hw.logicalcpu)

# Verify build
BINARY_PATH="$BUILD_DIR/bin/llama-server"

if [ -f "$BINARY_PATH" ]; then
    echo ""
    echo "✅ SUCCESS! llama-server is built."
    echo "📍 Binary location: $(pwd)/$BINARY_PATH"
    echo ""
    echo "👉 To run the server, use: ./start_server_parallel.sh"
else
    echo "❌ Compilation failed. Binary not found at $BINARY_PATH"
    exit 1
fi