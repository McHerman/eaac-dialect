func.func private @add(%arg0: tensor<128x128xi8>, %arg1: tensor<128x128xi8>) -> tensor<128x128xi8> {
  %cst = arith.constant 0 : i8
  %init = tensor.empty() : tensor<128x128xi8>
  %fill = linalg.fill ins(%cst : i8) outs(%init : tensor<128x128xi8>) -> tensor<128x128xi8>

  %s0 = linalg.add ins(%arg0, %arg1 : tensor<128x128xi8>, tensor<128x128xi8>)
                   outs(%fill : tensor<128x128xi8>) -> tensor<128x128xi8>
  return %s0 : tensor<128x128xi8>
}

func.func @main() -> tensor<128x128xi8> {
  //%cst = arith.constant 0 : i8
  %c1 = arith.constant 1 : i8
  %c2 = arith.constant 2 : i8
  %c3 = arith.constant 3 : i8

  %a_empty = tensor.empty() : tensor<128x128xi8>
  %a_full = linalg.fill ins(%c1 : i8) outs(%a_empty : tensor<128x128xi8>) -> tensor<128x128xi8>

  %b_empty = tensor.empty() : tensor<128x128xi8>
  %b_full = linalg.fill ins(%c2 : i8) outs(%b_empty : tensor<128x128xi8>) -> tensor<128x128xi8>

  %c_empty = tensor.empty() : tensor<128x128xi8>
  %c_full = linalg.fill ins(%c3 : i8) outs(%b_empty : tensor<128x128xi8>) -> tensor<128x128xi8>

  %s0 = func.call @add(%a_full, %b_full) : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi8>
  %s1 = func.call @add(%s0, %c_full) : (tensor<128x128xi8>, tensor<128x128xi8>) -> tensor<128x128xi8>

  return %s1 : tensor<128x128xi8>
}
