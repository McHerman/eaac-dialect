// GeMM test function using linalg.matmul
// Input A: [64, 128] (M=64, K=128)
// Input B: [128, 32] (K=128, N=32)
// Output:  [64, 32]  (M=64, N=32)

func.func private @gemm(%arg0: tensor<64x128xf32>, %arg1: tensor<128x32xf32>) -> tensor<64x32xf32> {
  %cst = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<64x32xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%init : tensor<64x32xf32>) -> tensor<64x32xf32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<128x32xf32>)
                     outs(%fill : tensor<64x32xf32>) -> tensor<64x32xf32>
  return %0 : tensor<64x32xf32>
}

func.func @main() -> tensor<64x32xf32> {
  %cst = arith.constant 1.0 : f32

  // Create input tensors
  %a_init = tensor.empty() : tensor<64x128xf32>
  %a = linalg.fill ins(%cst : f32) outs(%a_init : tensor<64x128xf32>) -> tensor<64x128xf32>

  %b_init = tensor.empty() : tensor<128x32xf32>
  %b = linalg.fill ins(%cst : f32) outs(%b_init : tensor<128x32xf32>) -> tensor<128x32xf32>

  // Call gemm function
  %result = func.call @gemm(%a, %b) : (tensor<64x128xf32>, tensor<128x32xf32>) -> tensor<64x32xf32>

  return %result : tensor<64x32xf32>
}
