# EAAC Dialect

An out-of-tree MLIR dialect and compiler pipeline that lowers tensor/linalg
programs into scheduled, async accelerator instructions, emitted as
`.eaac` FlatBuffer binaries for the Scala simulation fabric. Also supports
lowering to RISC-V/LLVM IR.

## Prerequisites

- A built LLVM/MLIR tree (see `flake.nix` / `llvm_build_dir` in `justfile`)
- CMake 3.20+, Ninja, C++17 compiler, [`just`](https://github.com/casey/just)

## Building

```bash
just configure   # cmake configure against your llvm-project build
just build
```

## Usage

```bash
./build/bin/eaac-opt <input.mlir> [passes...]
just help-passes            # list available passes
just test                   # run lit/FileCheck regression tests
just demo                   # run the demo pipeline end-to-end
just to-json <name>         # inspect a .eaac binary as JSON
```

## Layout

- `include/eaac/`, `lib/` – dialect ops, types, passes, transforms
- `eaac-opt.cpp` – `mlir-opt`-like driver
- `schema.fbs` – FlatBuffer schema for the `.eaac` output format
- `test/` – lit tests
