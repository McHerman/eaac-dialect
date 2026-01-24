#!/usr/bin/env python3
"""
Simple EAAC MLIR generator using text templates.
This generates MLIR text that can be passed to eaac-opt.
"""

def generate_constant_tensor(name, values, shape):
    """Generate an EAAC constant operation."""
    flat_values = ", ".join(str(v) for v in values)
    shape_str = "x".join(str(d) for d in shape)

    return f"""
  func.func @{name}() -> tensor<{shape_str}xf64> {{
    %0 = eaac.constant dense<{flat_values}> : tensor<{shape_str}xf64>
    return %0 : tensor<{shape_str}xf64>
  }}
"""

def generate_add_function(name, shape):
    """Generate an EAAC addition function."""
    shape_str = "x".join(str(d) for d in shape)
    tensor_type = f"tensor<{shape_str}xf64>"

    return f"""
  func.func @{name}(%arg0: {tensor_type}, %arg1: {tensor_type}) -> {tensor_type} {{
    %0 = eaac.add %arg0, %arg1 : {tensor_type}
    return %0 : {tensor_type}
  }}
"""

def generate_matmul_function(name, m, n, k):
    """Generate an EAAC matrix multiplication function."""
    return f"""
  func.func @{name}(%arg0: tensor<{m}x{k}xf64>, %arg1: tensor<{k}x{n}xf64>) -> tensor<{m}x{n}xf64> {{
    %0 = eaac.generic_call @multiply_transpose(%arg0, %arg1) : (tensor<{m}x{k}xf64>, tensor<{k}x{n}xf64>) -> tensor<{m}x{n}xf64>
    return %0 : tensor<{m}x{n}xf64>
  }}
"""

def generate_module(functions):
    """Wrap functions in a module."""
    functions_str = "\n".join(functions)
    return f"""module {{{functions_str}
}}
"""

if __name__ == "__main__":
    # Example: Generate some EAAC MLIR
    functions = [
        generate_constant_tensor("make_matrix", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]),
        generate_add_function("add_matrices", [2, 3]),
        generate_matmul_function("matmul", 4, 4, 3),
    ]

    mlir_code = generate_module(functions)

    # Write to file
    with open("generated.mlir", "w") as f:
        f.write(mlir_code)

    print("Generated MLIR written to generated.mlir")
    print("\nRun with: ./build/bin/eaac-opt generated.mlir")
