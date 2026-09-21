module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 2048, 4096, 4096>,
      //"tier_capacities" = array<i64: 65536, 65536, 65536>,
      "alias_check_length" = 4096 : i64,
      "reuse_guard" = 100000 : i64,
      "num_semaphore_pairs" = 8 : i64,
      "num_semaphore_generations" = 4 : i64,
      "bus_size" = 64 : i64
    >
  >
} {
  eaac.schedule @schedule {
    eaac.hw_alloc @unit0 : !eaac.gemm<2 32>
  }

  func.func @main(%arg0: tensor<16x16xi8>) -> tensor<16x16xi32> {
    %a = arith.constant dense<"0xe65c61df33bd6a4bdc83e2738e1575aeea74e33bc75419473c2ec11994ac4b18e65670a7f998a2f252c16fa756741799ca115e8e4ad511f5d7383d5df46dedd5e65c61df33bd6a4bdc83e2738e1575aeea74e33bc75419473c2ec11994ac4b18e65670a7f998a2f252c16fa756741799ca115e8e4ad511f5d7383d5df46dedd5e65c61df33bd6a4bdc83e2738e1575aeea74e33bc75419473c2ec11994ac4b18e65670a7f998a2f252c16fa756741799ca115e8e4ad511f5d7383d5df46dedd5e65c61df33bd6a4bdc83e2738e1575aeea74e33bc75419473c2ec11994ac4b18e65670a7f998a2f252c16fa756741799ca115e8e4ad511f5d7383d5df46dedd5">
        : tensor<16x16xi8>

    %cst = arith.constant 0 : i32
    %out = tensor.empty() : tensor<16x16xi32>
    %acc_7 = linalg.fill ins(%cst : i32) outs(%out : tensor<16x16xi32>) -> tensor<16x16xi32>
    %s0 = linalg.matmul ins(%a, %arg0 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%out : tensor<16x16xi32>) -> tensor<16x16xi32>
  
    return %s0 : tensor<16x16xi32>
  }
}
