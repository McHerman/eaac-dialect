module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      //"tier_capacities" = array<i64: 256, 2048, 16384>,
      "tier_capacities" = array<i64: 2048, 4096, 4096>,
      "alias_check_length" = 128 : i64,
      "reuse_guard" = 0 : i64,
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
    %w1  = arith.constant dense<"0x1a892cd0883713c2aaae1a5145acaa7d5300df2cc12303a1e534162f48400a95dcf487f0129caff769a106cbdef8d6bf105c2af98657abacc825e9c6ac08602d">
        : tensor<8x8xi8>
    %w2  = arith.constant dense<"0xba06b475cf1e93fd77f75a4ee3c1ec160544e9620a83b48bded7003720b52d2add04c125ce0ea586f110abd4b20c53d62049be78281dc69c4a95c7bc40ac3d55">
        : tensor<8x8xi8>
    %w3  = arith.constant dense<"0xc5dd1f45e80ab2e6e69bdb0e51a646af35bc73ae2f8b7c68a487cb16296466086389762ba22aae4445f7fccb0e83f8aeca724003fdaf3e1ab8ebb9d7a1b76ee5">
        : tensor<8x8xi8>
    %w4  = arith.constant dense<"0xaf625cc31fd4c08759c0bc23973e87a41cc21ac69837eea67ce488e0b07fe98cac185d3f0adbfe43f415f048a6bd7110ed9307a4c57420819156d5b7d91606fe">
        : tensor<8x8xi8>

    %s0 = func.call @matmul(%arg0, %arg1) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>

    %s1 = func.call @matmul(%w1, %s0) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %s2 = func.call @matmul(%w2, %s0) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %s3 = func.call @matmul(%w3, %s0) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %s4 = func.call @matmul(%w4, %s0) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>

    %s5 = func.call @matmul(%s1, %s3) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %s6 = func.call @matmul(%s2, %s4) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %s7 = func.call @matmul(%s5, %s6) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>

    return %s7 : tensor<8x8xi8>
  }
}
