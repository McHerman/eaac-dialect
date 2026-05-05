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

test-buf:
    {{eaac_opt}} --inline --one-shot-bufferize="bufferize-function-boundaries"  test/input_8_tiny.mlir

test-local-staging:
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --inline --canonicalize --eaac-local-staging -debug-only=local-staging test/input_8_tiny.mlir

test-insert:
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --inline --canonicalize --eaac-insert-load-store test/test_load_insert.mlir

test-almost-full:
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --inline --canonicalize --eaac-insert-load-store --eaac-local-staging --eaac-lower-copy-to-dma --eaac-encode-dependencies --eaac-find-async-dependency --eaac-insert-require --eaac-lower-async-to-semaphore --eaac-assign-semaphore-addresses --convert-linalg-to-eaac test/input_8_tiny.mlir

test-full input="input_8_tiny":
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --inline --canonicalize --eaac-insert-load-store --eaac-local-staging --eaac-lower-copy-to-dma --eaac-encode-dependencies --eaac-find-async-dependency --eaac-insert-require --eaac-lower-async-to-semaphore --eaac-assign-semaphore-addresses --convert-linalg-to-eaac --mlir-print-ir-after-all test/{{input}}.mlir

# Configure and build with -O0 for debugging (separate build dir)
build-debug:
    mkdir -p {{debug_build_dir}}
    cmake -G Ninja -S . -B {{debug_build_dir}} \
        -DMLIR_DIR="{{llvm_build_dir}}/lib/cmake/mlir" \
        -DLLVM_DIR="{{llvm_build_dir}}/lib/cmake/llvm" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLLVM_ENABLE_LLD=ON
    cmake --build {{debug_build_dir}}

# Run lit/FileCheck regression tests
test: build
    {{llvm_build_dir}}/bin/llvm-lit {{build_dir}}/test -v

# Run full pipeline and serialize to FlatBuffer binary
translate input="input_8_tiny":
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --inline --canonicalize --eaac-insert-load-store --eaac-local-staging --eaac-lower-copy-to-dma --eaac-encode-dependencies --eaac-find-async-dependency --eaac-insert-require --eaac-lower-async-to-semaphore --eaac-assign-semaphore-addresses --convert-linalg-to-eaac test/{{input}}.mlir | {{build_dir}}/bin/eaac-translate --eaac-to-flatbuffer -o {{input}}.eaac
    flatc --json --raw-binary include/eaac/Target/eaac_program.fbs -- {{input}}.eaac
    python tools/test-harness/reference_runner.py test/{{input}}.mlir -o {{input}}.reference.json

# Rebuild FlatBuffer schema header
flatbuf:
    cmake --build {{build_dir}} --target EAACFlatBufferGen

# Print available passes
help-passes:
    {{eaac_opt}} --help | grep -A 1000 "Passes:"
