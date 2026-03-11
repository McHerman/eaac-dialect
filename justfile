# EAAC Dialect Build and Run Commands

# Configuration
llvm_build_dir := "/home/karlhk/dtu/Thesis/MLIR/llvm-project/build"
build_dir := "build"
build_type := "RelWithDebInfo"
eaac_opt := build_dir / "bin/eaac-opt"

# Default recipe - show available commands
default:
    @just --list

# Configure the build with CMake
configure:
    mkdir -p {{build_dir}}
    cmake -G Ninja -S . -B {{build_dir}} \
        -DMLIR_DIR="{{llvm_build_dir}}/lib/cmake/mlir" \
        -DLLVM_DIR="{{llvm_build_dir}}/lib/cmake/llvm" \
        -DCMAKE_BUILD_TYPE={{build_type}} \
        -DLLVM_ENABLE_LLD=ON

# Build the project
build:
    cmake --build {{build_dir}}

# Configure and build
all: configure build

# Clean build artifacts
clean:
    rm -rf {{build_dir}}

# Rebuild from scratch
rebuild: clean all

# Run the test input file with bufferization
test:
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --buffer-results-to-out-params --buffer-deallocation-pipeline test/input.mlir

# Run with collect-alloc-dealloc pass
test-collect:
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --buffer-results-to-out-params --buffer-deallocation-pipeline --eaac-collect-alloc-dealloc test/input.mlir

test-alloc:
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --buffer-results-to-out-params --buffer-deallocation-pipeline --eaac-memory-alloc test/input.mlir


test-local-staging:
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --buffer-results-to-out-params --buffer-deallocation-pipeline --inline --canonicalize --eaac-local-staging test/input.mlir

# Print available passes
help-passes:
    {{eaac_opt}} --help | grep -A 1000 "Passes:"
