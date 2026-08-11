# EAAC Dialect Build and Run Commands

# Configuration
llvm_build_dir := "/home/karlhk/dtu/Thesis/MLIR/llvm-project/build"
dir := "/home/karlhk/dtu/Thesis/MLIR/eaac-dialect"
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

test-almost-full input="riscv-extrasmall":
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" \
    --inline \
    --canonicalize \
    --eaac-insert-load-store \
    --eaac-local-staging \
    --eaac-lower-copy-to-dma \
    --eaac-encode-dependencies \
    --eaac-find-async-dependency \
    --eaac-insert-require \
    --eaac-correct-broadcast \
    --eaac-find-alias-dependency \
    --eaac-lower-async-to-semaphore \
    --eaac-assign-semaphore-addresses \
    "--transform-preload-library=transform-library-paths=test/linalg-to-eaac.transform.mlir" \
    --transform-interpreter \
    --eaac-legalize-for-hw \
    --eaac-riscv-kernel-to-function \
    test/{{input}}.mlir
    #--eaac-riscv-kernel-to-llvm \
    #--eaac-lower-memref-to-llvm \
    #--eaac-split-llvm-from-eaac=llvm-output-file={{dir}}/{{input}}.ll \

test-full input="input_8_tiny":
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" --inline --canonicalize --eaac-insert-load-store --eaac-local-staging --eaac-lower-copy-to-dma --eaac-encode-dependencies --eaac-find-async-dependency --eaac-insert-require --eaac-correct-broadcast --eaac-find-alias-dependency --eaac-lower-async-to-semaphore --eaac-assign-semaphore-addresses --convert-linalg-to-eaac --mlir-print-ir-after=eaac-find-alias-dependency -debug-only=find-alias-dependency test/{{input}}.mlir

test-single input pass:
    {{eaac_opt}} {{pass}} test/{{input}}.mlir

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


test-only input: build
    {{llvm_build_dir}}/bin/llvm-lit {{build_dir}}/test/{{input}} -v

translate input="input_8_tiny":
    {{eaac_opt}} --one-shot-bufferize="bufferize-function-boundaries" \
    --inline \
    --canonicalize \
    --eaac-insert-load-store \
    --eaac-local-staging \
    --eaac-lower-copy-to-dma \
    --eaac-encode-dependencies \
    --eaac-find-async-dependency \
    --eaac-insert-require \
    --eaac-correct-broadcast \
    --eaac-find-alias-dependency \
    --eaac-lower-async-to-semaphore \
    --eaac-assign-semaphore-addresses \
    "--transform-preload-library=transform-library-paths=test/linalg-to-eaac.transform.mlir" \
    --transform-interpreter \
    --eaac-legalize-for-hw \
    --eaac-riscv-kernel-to-function \
    --eaac-riscv-kernel-to-llvm \
    --eaac-lower-memref-to-llvm \
    --eaac-split-llvm-from-eaac=llvm-output-file={{dir}}/{{input}}.ll \
    test/{{input}}.mlir -o /tmp/{{input}}.eaac-stage.mlir
    {{build_dir}}/bin/eaac-translate --eaac-to-flatbuffer -o {{input}}.eaac /tmp/{{input}}.eaac-stage.mlir
    python tools/test-harness/reference_runner.py test/{{input}}.mlir -o {{input}}.reference.json
    cp {{input}}.eaac ../../hardware/ATAN/test
    cp {{input}}.reference.json ../../hardware/ATAN/test
    mkdir -p ../../hardware/ATAN/test/{{input}}
    cp {{input}}.ll ../../hardware/ATAN/test/{{input}}/{{input}}.ll

translate-riscv input="ad-bottleneck-8tile-baseline":
    {{eaac_opt}} \
    --one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" \
    --buffer-results-to-out-params="hoist-static-allocs modify-public-functions" \
    --promote-buffers-to-stack="max-alloc-size-in-bytes=4096" \
    --inline --canonicalize \
    --convert-linalg-to-loops --lower-affine --convert-scf-to-cf \
    --convert-arith-to-llvm \
    --convert-func-to-llvm="use-bare-ptr-memref-call-conv" \
    --finalize-memref-to-llvm \
    --convert-cf-to-llvm --convert-index-to-llvm --reconcile-unrealized-casts \
    test/{{input}}.mlir -o /tmp/{{input}}.baseline-stage.mlir
    {{llvm_build_dir}}/bin/mlir-translate --mlir-to-llvmir /tmp/{{input}}.baseline-stage.mlir \
    -o ../../hardware/ATAN/test/{{input}}/{{input}}.ll

# Convert a .eaac flatbuffer binary to JSON using the schema
to-json input="input_8_tiny":
    flatc --json --raw-binary include/eaac/Target/eaac_program.fbs -- {{input}}.eaac

# Rebuild FlatBuffer schema header
flatbuf:
    cmake --build {{build_dir}} --target EAACFlatBufferGen

# Print available passes
help-passes:
    {{eaac_opt}} --help | grep -A 1000 "Passes:"
