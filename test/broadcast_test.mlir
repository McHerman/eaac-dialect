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
// RUN: --eaac-lower-async-to-semaphore \
// RUN: --eaac-assign-semaphore-addresses \

// RUN:   | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK: linalg.matmul

// CHECK: %sem
// CHECK: %sem_11 = eaac.sem_alloc(%c0, %c64) chains_from(%sem : !eaac.semaphore<addr = 0, gen = 0>) event_mode = r {eaac.sem_addr = 6 : index, eaac.sem_gen = 0 : index} : <addr = 6, gen = 0>
// CHECK: %sem_18 = eaac.sem_alloc(%c0, %c64) chains_from(%sem : !eaac.semaphore<addr = 0, gen = 0>) event_mode = r {eaac.sem_addr = 10 : index, eaac.sem_gen = 0 : index} : <addr = 10, gen = 0>

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

  func.func private @matmul(%arg0: tensor<8x8xi8>, %arg1: tensor<8x8xi8>) -> tensor<8x8xi8> {
    %cst = arith.constant 0 : i8
    %init = tensor.empty() : tensor<8x8xi8>
    %fill = linalg.fill ins(%cst : i8) outs(%init : tensor<8x8xi8>) -> tensor<8x8xi8>
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<8x8xi8>, tensor<8x8xi8>)
                       outs(%fill : tensor<8x8xi8>) -> tensor<8x8xi8>
    return %0 : tensor<8x8xi8>
  }

  func.func @main(%x: tensor<8x8xi8>) -> tensor<8x8xi8> {
    %w1 = arith.constant dense<"0x1e652ec152d0bfcd65190ffc604c0933d0423381de26c82d9cb255cc124c306c02ddac6646e36f131ebdde5d3ab46f152d77b8f00ae71c414b3ef38784310f72"> : tensor<8x8xi8>
    %w2 = arith.constant dense<"0xd77149428ef7412fde187b7314d70e42539fee8f28e1942433db2d23c8a04fd523fd2b5b15312bda21f718fc890d1f6a5c382f2a7905791976ffa7cbf1509065"> : tensor<8x8xi8>
    %w3 = arith.constant dense<"0xfeb871ede4e8ce0dab21f3e72c97ad10741fa3750047c8af107bc51f89cf4382c1da646583e8654445805c0cc267aee4ba4cdec631651edc5c8a7a20e7ddd9a5"> : tensor<8x8xi8>
    %w4 = arith.constant dense<"0xd736606bae1f4dbd237803490d414e2fa6dc9b0f3cda3df519970f7e08c792e31d5ff9d8fbc9e15478ef4358f158500aed4eb60fa64b9fe1c7fada71f9dc32e0"> : tensor<8x8xi8>

    %h1 = func.call @matmul(%x, %w1) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h2 = func.call @matmul(%x, %w2) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h3 = func.call @matmul(%x, %w3) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>

    %h4 = func.call @matmul(%h1, %h2) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h5 = func.call @matmul(%h4, %w3) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
  
    return %h5 : tensor<8x8xi8>
  }
}

