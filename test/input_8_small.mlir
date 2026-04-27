func.func private @gemm(%arg0: tensor<128x128xi8>, %arg1: tensor<128x128xi8>) -> tensor<128x128xi32> {
  %cst = arith.constant 0 : i32
  %init = tensor.empty() : tensor<128x128xi32>
  %fill = linalg.fill ins(%cst : i32) outs(%init : tensor<128x128xi32>) -> tensor<128x128xi32>
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<128x128xi8>, tensor<128x128xi8>)
                     outs(%fill : tensor<128x128xi32>) -> tensor<128x128xi32>
  return %0 : tensor<128x128xi32>
}

func.func @main(%a: tensor<128x128xi8>) -> tensor<128x128xi32> {
  %c1 = arith.constant 1 : i8
  %c2 = arith.constant 2 : i8
  %c3 = arith.constant 3 : i8

  %b_init = tensor.empty() : tensor<128x128xi8>
  %b = linalg.fill ins(%c2 : i8) outs(%b_init : tensor<128x128xi8>) -> tensor<128x128xi8>

  %c_init = tensor.empty() : tensor<128x128xi8>
  %c = linalg.fill ins(%c3 : i8) outs(%c_init : tensor<128x128xi8>) -> tensor<128x128xi8>

  // R0 = A * B  (reuses A, B)
  %r0 = func.call @gemm(%a, %b) : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>

  // R1 = A * C  (reuses A)
  %r1 = func.call @gemm(%a, %c) : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi32>

  return %r1 : tensor<128x128xi32>
}
