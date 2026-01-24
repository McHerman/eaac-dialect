#!/usr/bin/env bash
set -e

# Path to your LLVM/MLIR build directory
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-../llvm-project/build}"

if [ ! -d "$LLVM_BUILD_DIR" ]; then
    echo "Error: LLVM build directory not found at $LLVM_BUILD_DIR"
    echo "Please set LLVM_BUILD_DIR environment variable to your LLVM build directory"
    echo "Example: export LLVM_BUILD_DIR=/path/to/llvm-project/build"
    exit 1
fi

# Create build directory
mkdir -p build
cd build

# Configure
cmake -G Ninja .. \
    -DMLIR_DIR="$LLVM_BUILD_DIR/lib/cmake/mlir" \
    -DLLVM_DIR="$LLVM_BUILD_DIR/lib/cmake/llvm" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_LLD=ON

# Build
ninja
