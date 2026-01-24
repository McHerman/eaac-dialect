module {
  func.func @make_matrix() -> tensor<2x3xf64> {
    %0 = eaac.constant dense<1.0, 2.0, 3.0, 4.0, 5.0, 6.0> : tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }


  func.func @add_matrices(%arg0: tensor<2x3xf64>, %arg1: tensor<2x3xf64>) -> tensor<2x3xf64> {
    %0 = eaac.add %arg0, %arg1 : tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }


  func.func @matmul(%arg0: tensor<4x3xf64>, %arg1: tensor<3x4xf64>) -> tensor<4x4xf64> {
    %0 = eaac.generic_call @multiply_transpose(%arg0, %arg1) : (tensor<4x3xf64>, tensor<3x4xf64>) -> tensor<4x4xf64>
    return %0 : tensor<4x4xf64>
  }

}
