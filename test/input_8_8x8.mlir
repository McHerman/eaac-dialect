// 8x8 version of input_8_16x16.mlir
// Generated with random i8 values (seed=42).

module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 2048, 4096, 4096>,
      //"tier_capacities" = array<i64: 65536, 65536, 65536>,
      "alias_check_length" = 4096 : i64,
      "reuse_guard" = 100000 : i64,
      "num_semaphore_pairs" = 8 : i64,
      "num_semaphore_generations" = 4 : i64,
      "bus_size" = 8 : i64
    >
  >
} {
  eaac.schedule @schedule {
    eaac.hw_alloc @unit0 : !eaac.gemm<2 32>
  }

  func.func private @matmul(%arg0: tensor<8x8xi8>, %arg1: tensor<8x8xi8>) -> tensor<8x8xi8> {
    %cst = arith.constant 0 : i8
    %init = tensor.empty() : tensor<8x8xi8>
    %fill = linalg.fill ins(%cst : i8) outs(%init : tensor<8x8xi8>) -> tensor<8x8xi8>
  
    %s0 = linalg.matmul ins(%arg0, %arg1 : tensor<8x8xi8>, tensor<8x8xi8>)
                        outs(%fill : tensor<8x8xi8>) -> tensor<8x8xi8>
    return %s0 : tensor<8x8xi8>
  }
  
  func.func @main(%arg0: tensor<8x8xi8>, %arg1: tensor<8x8xi8>) -> tensor<8x8xi8> {
    %a = arith.constant dense<"0xe65c61df33bd6a4bdc83e2738e1575aeea74e33bc75419473c2ec11994ac4b18e65670a7f998a2f252c16fa756741799ca115e8e4ad511f5d7383d5df46dedd5">
        : tensor<8x8xi8>
    %b = arith.constant dense<"0xe32c6219e76b12a417bec435022020261586c585b47df08e8183cc78d76954386b759a551d7dc470a55bdbb60189b3803f940cae3b13817e941973ae20c4931e">
        : tensor<8x8xi8>
  
    %s0 = func.call @matmul(%a, %arg0) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %s1 = func.call @matmul(%b, %arg1) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
  
    %s3 = func.call @matmul(%s0, %s1) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
  
    return %s3 : tensor<8x8xi8>
  }
}
