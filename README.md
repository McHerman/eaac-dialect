# EAAC Dialect

An out-of-tree MLIR dialect for MLIR-only transformations (no custom parser/frontend).

This dialect is based on the Toy tutorial but simplified to work only with MLIR input files,
similar to `mlir-opt`. It includes:
- Custom dialect and operations
- Shape inference
- Lowering to Affine loops
- Lowering to LLVM IR
- Optimization passes

## Prerequisites

- A built version of LLVM/MLIR (see `../llvm-project/`)
- CMake 3.20+
- Ninja build system
- C++17 compatible compiler

## Building

### Using the build script

```bash
# The script assumes LLVM is built at ../llvm-project/build
./build.sh
```

### Manual CMake configuration

```bash
mkdir build && cd build

cmake -G Ninja .. \
    -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
    -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo

ninja
```

## Using the EAAC Dialect

The dialect provides an `eaac-opt` tool similar to `mlir-opt` for testing and transforming MLIR code:

```bash
./build/bin/eaac-opt <input.mlir> [options]
```

### Example MLIR Code

Create a file `test.mlir`:

```mlir
module {
  func.func @test_constant() -> tensor<2x3xf64> {
    %0 = eaac.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}
```

Run it through eaac-opt:

```bash
./build/bin/eaac-opt test.mlir
```

### Available Operations

The EAAC dialect includes operations from the Toy tutorial:
- `eaac.constant` - Constant tensor values
- `eaac.add` - Element-wise addition
- `eaac.mul` - Element-wise multiplication
- `eaac.transpose` - Transpose operation
- `eaac.reshape` - Reshape operation
- `eaac.generic_call` - Generic function call

See `include/eaac/Ops.td` for complete operation definitions.

## Project Structure

```
eaac-dialect/
├── include/eaac/          # Header files and TableGen definitions
│   ├── Dialect.h          # Dialect definition
│   ├── Ops.td             # Operation TableGen definitions
│   ├── Passes.h           # Pass declarations
│   └── ShapeInferenceInterface.td  # Shape inference interface
├── mlir/                  # MLIR-specific implementation
│   ├── Dialect.cpp        # Dialect implementation
│   ├── LowerToAffineLoops.cpp  # Affine lowering pass
│   ├── LowerToLLVM.cpp    # LLVM IR lowering pass
│   ├── ShapeInferencePass.cpp  # Shape inference pass
│   └── EAACCombine.cpp    # Optimization passes
├── eaac-opt.cpp          # Main optimizer driver (similar to mlir-opt)
├── CMakeLists.txt         # Build configuration
├── build.sh               # Build helper script
└── README.md              # This file
```

## Adding New Operations

1. Define the operation in `include/eaac/Ops.td`
2. Rebuild to generate C++ code
3. Add any custom verifiers/builders in `mlir/Dialect.cpp`
4. Test with `eaac-opt`

Example operation definition in TableGen:

```tablegen
def EAAC_AddOp : EAAC_Op<"add", [Pure]> {
  let summary = "element-wise addition operation";
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
}
```

## Adding New Passes

1. Declare the pass in `include/eaac/Passes.h`
2. Implement the pass in a new file under `mlir/`
3. Update `CMakeLists.txt` to include the new file
4. Use via command line: `eaac-opt --pass-name <input.mlir>`

## References

- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [MLIR Documentation](https://mlir.llvm.org/)
- [TableGen Documentation](https://llvm.org/docs/TableGen/)
