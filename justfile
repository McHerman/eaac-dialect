# EAAC Dialect Build and Run Commands

# Configuration
llvm_build_dir := "/home/karlhk/dtu/Thesis/MLIR/llvm-project/build"
build_dir := "build/bin"
build_type := "RelWithDebInfo"

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

# Run eaac-opt on a file
test:
    #{{build_dir}}/eaac-opt --one-shot-bufferize="bufferize-function-boundaries" --buffer-deallocation-pipeline --eaac-collect-alloc-dealloc test/input.mlir
    {{build_dir}}/eaac-opt --one-shot-bufferize="bufferize-function-boundaries" --buffer-results-to-out-params --buffer-deallocation-pipeline --buffer-deallocation-simplification test/input.mlir 
