// Simple test file for EAAC dialect
module {
  func.func @test_constant() -> tensor<2x3xf64> {
    %0 = eaac.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    return %0 : tensor<2x3xf64>
  }
}
