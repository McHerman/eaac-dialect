// Multi-GEMM test: reuses input tensors across multiple matmuls
// All tensors are 128x128, inputs i8, accumulators i32

func.func private @gemm(%arg0: tensor<128x128xi8>, %arg1: tensor<128x128xi8>) -> tensor<128x128xi32> {
  %cst = arith.constant 0 : i32
  %init = tensor.empty() : tensor<128x128xi32>
  %fill = linalg.fill ins(%cst : i32) outs(%init : tensor<128x128xi32>) -> tensor<128x128xi32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xi8>, tensor<128x128xi8>)
                     outs(%fill : tensor<128x128xi32>) -> tensor<128x128xi32>
  return %0 : tensor<128x128xi32>
}

func.func @main() -> tensor<128x128xi32> {
  %c1 = arith.constant 1 : i8
  %c2 = arith.constant 2 : i8
  %c3 = arith.constant 3 : i8

  // Create input tensors A, B, C, D
  %a_init = tensor.empty() : tensor<128x128xi8>
  %a = linalg.fill ins(%c1 : i8) outs(%a_init : tensor<128x128xi8>) -> tensor<128x128xi8>

  %b_init = tensor.empty() : tensor<128x128xi8>
  %b = linalg.fill ins(%c2 : i8) outs(%b_init : tensor<128x128xi8>) -> tensor<128x128xi8>

  %c_init = tensor.empty() : tensor<128x128xi8>
  %c = linalg.fill ins(%c3 : i8) outs(%c_init : tensor<128x128xi8>) -> tensor<128x128xi8>

  %d_init = tensor.empty() : tensor<128x128xi8>
  %d = linalg.fill ins(%c1 : i8) outs(%d_init : tensor<128x128xi8>) -> tensor<128x128xi8>

  // R0 = A * B  (reuses A, B)
  %r0 = func.call @gemm(%a, %b) : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>

  // R1 = A * C  (reuses A)
  %r1 = func.call @gemm(%a, %c) : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>

  // R2 = B * D  (reuses B)
  %r2 = func.call @gemm(%b, %d) : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>

  // R3 = C * D  (reuses C, D)
  %r3 = func.call @gemm(%c, %d) : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>

  // R4 = C * B  (reuses C, B)
  %r4 = func.call @gemm(%c, %b) : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>

  // Combine results: sum = R0 + R1 + R2 + R3 + R4
  %s0 = linalg.add ins(%r0, %r1 : tensor<128x128xi32>, tensor<128x128xi32>)
                   outs(%r0 : tensor<128x128xi32>) -> tensor<128x128xi32>
  %s1 = linalg.add ins(%s0, %r2 : tensor<128x128xi32>, tensor<128x128xi32>)
                   outs(%s0 : tensor<128x128xi32>) -> tensor<128x128xi32>
  %s2 = linalg.add ins(%s1, %r3 : tensor<128x128xi32>, tensor<128x128xi32>)
                   outs(%s1 : tensor<128x128xi32>) -> tensor<128x128xi32>
  %s3 = linalg.add ins(%s2, %r4 : tensor<128x128xi32>, tensor<128x128xi32>)
                   outs(%s2 : tensor<128x128xi32>) -> tensor<128x128xi32>

  return %s3 : tensor<128x128xi32>
}
