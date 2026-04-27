func.func @main(%a: tensor<128x128xi8>, %b: tensor<128x128xi8>) -> tensor<128x128xi8> {

  %cst = arith.constant 0 : i8
  %init = tensor.empty() : tensor<128x128xi8>
  %fill = linalg.fill ins(%cst : i8) outs(%init : tensor<128x128xi8>) -> tensor<128x128xi8>

  %s0 = linalg.matmul ins(%a, %b : tensor<128x128xi8>, tensor<128x128xi8>)
                      outs(%fill : tensor<128x128xi8>) -> tensor<128x128xi8>

  return %s0 : tensor<128x128xi8>
}
