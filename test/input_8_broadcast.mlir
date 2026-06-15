module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 1024, 2048, 16384>,
      "alias_check_length" = 128 : i64,
      "num_semaphore_pairs" = 16 : i64,
      "num_semaphore_generations" = 4 : i64
    >
  >
} {
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

    %s0 = func.call @matmul(%arg0, %arg1) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>

    %s1 = func.call @matmul(%a, %s0) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %s2 = func.call @matmul(%b, %s0) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
  
    %s3 = func.call @matmul(%s1, %s2) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
  
    return %s3 : tensor<8x8xi8>
  }
}
