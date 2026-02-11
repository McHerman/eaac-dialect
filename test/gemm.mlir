// Test file for GEMM operation with bufferization and liveness analysis
//
// Run with:
//   eaac-opt test/gemm.mlir \
//     -one-shot-bufferize="bufferize-function-boundaries" \
//     -eaac-collect-alloc-dealloc
//
// This will:
//   1. Bufferize the tensor operations to memref operations
//   2. Run the collect alloc/dealloc pass to print liveness info

module {
  // Simple GEMM: C = A * B
  // A: 4x8, B: 8x16, C: 4x16
  func.func @gemm(%A: tensor<4x8xf32>, %B: tensor<8x16xf32>) -> tensor<4x16xf32> {
    // Initialize output tensor with zeros
    %cst = arith.constant 0.0 : f32
    %C_init = tensor.empty() : tensor<4x16xf32>
    %C_zero = linalg.fill ins(%cst : f32) outs(%C_init : tensor<4x16xf32>) -> tensor<4x16xf32>

    // Perform matrix multiplication: C = A @ B
    %C = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x16xf32>)
                       outs(%C_zero : tensor<4x16xf32>) -> tensor<4x16xf32>

    return %C : tensor<4x16xf32>
  }

  // GEMM with multiple intermediate buffers for liveness demonstration
  func.func @gemm_chain(%A: tensor<4x8xf32>, %B: tensor<8x8xf32>, %C: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %cst = arith.constant 0.0 : f32

    // First matmul: tmp = A @ B (4x8 @ 8x8 = 4x8)
    %tmp_init = tensor.empty() : tensor<4x8xf32>
    %tmp_zero = linalg.fill ins(%cst : f32) outs(%tmp_init : tensor<4x8xf32>) -> tensor<4x8xf32>
    %tmp = linalg.matmul ins(%A, %B : tensor<4x8xf32>, tensor<8x8xf32>)
                         outs(%tmp_zero : tensor<4x8xf32>) -> tensor<4x8xf32>

    // Second matmul: result = tmp @ C (4x8 @ 8x16 = 4x16)
    %result_init = tensor.empty() : tensor<4x16xf32>
    %result_zero = linalg.fill ins(%cst : f32) outs(%result_init : tensor<4x16xf32>) -> tensor<4x16xf32>
    %result = linalg.matmul ins(%tmp, %C : tensor<4x8xf32>, tensor<8x16xf32>)
                            outs(%result_zero : tensor<4x16xf32>) -> tensor<4x16xf32>

    return %result : tensor<4x16xf32>
  }
}
