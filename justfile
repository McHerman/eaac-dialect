# EAAC Dialect Build and Run Commands

# Configuration
llvm_build_dir := "/home/karlhk/dtu/Thesis/MLIR/llvm-project/build"
build_dir := "build"
debug_build_dir := "build-debug"
build_type := "RelWithDebInfo"
eaac_opt := build_dir / "bin/eaac-opt"
eaac_opt_debug := debug_build_dir / "bin/eaac-opt"

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

test-local-staging:
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --buffer-results-to-out-params --inline --canonicalize --eaac-local-staging -debug-only=local-staging --view-op-graph test/input_8_tiny.mlir

# Configure and build with -O0 for debugging (separate build dir)
build-debug:
    mkdir -p {{debug_build_dir}}
    cmake -G Ninja -S . -B {{debug_build_dir}} \
        -DMLIR_DIR="{{llvm_build_dir}}/lib/cmake/mlir" \
        -DLLVM_DIR="{{llvm_build_dir}}/lib/cmake/llvm" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_ENABLE_LLD=ON
    cmake --build {{debug_build_dir}}

# Debug with lldb (uses -O0 build so variables aren't optimized out)
test-debug:
    lldb -s debug.lldb -- {{eaac_opt_debug}} \
        --one-shot-bufferize="bufferize-function-boundaries" \
        --buffer-results-to-out-params --inline --canonicalize \
        --eaac-local-staging test/input_8_tiny.mlir

# Run lit/FileCheck regression tests
check: build
    {{llvm_build_dir}}/bin/llvm-lit {{build_dir}}/test -v

# Print available passes
help-passes:
    {{eaac_opt}} --help | grep -A 1000 "Passes:"
