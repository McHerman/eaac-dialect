// RUN: %eaac-opt %s \
// RUN: --one-shot-bufferize="bufferize-function-boundaries" \
// RUN: --inline \
// RUN: --canonicalize \
// RUN: --eaac-insert-load-store \
// RUN: --eaac-local-staging \
// RUN: --eaac-lower-copy-to-dma \
// RUN: --eaac-encode-dependencies \
// RUN: --eaac-find-async-dependency \
// RUN: --eaac-insert-require \
// RUN: --eaac-correct-broadcast \
// RUN: --eaac-find-alias-dependency \
// RUN: --eaac-lower-async-to-semaphore \
// RUN: --eaac-schedule \
// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK: linalg.matmul {hw_unit = @schedule::@unit0}

module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 16384, 32768, 131072>,
      "alias_check_length" = 128 : i64,
      "num_semaphore_pairs" = 16 : i64,
      "num_semaphore_generations" = 4 : i64
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
    %init = tensor.empty() : tensor<16x16xi32>
    %fill = linalg.fill ins(%cst : i32) outs(%init : tensor<16x16xi32>) -> tensor<16x16xi32>
  
    %s0 = linalg.matmul ins(%arg0, %a : tensor<16x16xi8>, tensor<16x16xi8>)
                        outs(%fill : tensor<16x16xi32>) -> tensor<16x16xi32>
  
    return %s0 : tensor<16x16xi32>
  }
}

