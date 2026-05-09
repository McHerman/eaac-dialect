// 10-layer MLP-style chain: each matmul has a single downstream consumer.
// H_{i+1} = H_i · W_{i+1}, with H_0 = X.
// All tensors 8x8 i8. Weights are constants (random, seed=123); X is an input
// argument. No SSA value is consumed by more than one downstream op, so this
// avoids the fanout case currently mishandled by the dependency analysis.

module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 1024, 2048, 16384>,
      "num_semaphore_pairs" = 8 : i64
    >
  >
} {
  func.func private @matmul(%arg0: tensor<8x8xi8>, %arg1: tensor<8x8xi8>) -> tensor<8x8xi8> {
    %cst = arith.constant 0 : i8
    %init = tensor.empty() : tensor<8x8xi8>
    %fill = linalg.fill ins(%cst : i8) outs(%init : tensor<8x8xi8>) -> tensor<8x8xi8>
  
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<8x8xi8>, tensor<8x8xi8>)
                       outs(%fill : tensor<8x8xi8>) -> tensor<8x8xi8>
    return %0 : tensor<8x8xi8>
  }
  
  func.func @main(%x: tensor<8x8xi8>) -> tensor<8x8xi8> {
    %w1  = arith.constant dense<"0x1a892cd0883713c2aaae1a5145acaa7d5300df2cc12303a1e534162f48400a95dcf487f0129caff769a106cbdef8d6bf105c2af98657abacc825e9c6ac08602d">
        : tensor<8x8xi8>
    %w2  = arith.constant dense<"0xba06b475cf1e93fd77f75a4ee3c1ec160544e9620a83b48bded7003720b52d2add04c125ce0ea586f110abd4b20c53d62049be78281dc69c4a95c7bc40ac3d55">
        : tensor<8x8xi8>
    %w3  = arith.constant dense<"0xc5dd1f45e80ab2e6e69bdb0e51a646af35bc73ae2f8b7c68a487cb16296466086389762ba22aae4445f7fccb0e83f8aeca724003fdaf3e1ab8ebb9d7a1b76ee5">
        : tensor<8x8xi8>
    %w4  = arith.constant dense<"0xaf625cc31fd4c08759c0bc23973e87a41cc21ac69837eea67ce488e0b07fe98cac185d3f0adbfe43f415f048a6bd7110ed9307a4c57420819156d5b7d91606fe">
        : tensor<8x8xi8>
    %w5  = arith.constant dense<"0x5465178954639bffef72902f34e28f6322c8d0e5381edddfb64efc44636c7b137d6a25ce0d05d3a40f157edf955f9b3990a66c0db94eee73d471dec860c727fe">
        : tensor<8x8xi8>
    %w6  = arith.constant dense<"0x159247f336403eebb408e3b493b96dc7183630a62a69c45cd640f34702532ca910324a70eadc66cf1e1c4d3aec2e4415b7e1697a78dcae492d0495a7cdb0751f">
        : tensor<8x8xi8>
    %w7  = arith.constant dense<"0x1d21387f9378400086d7bcbd42d51194b3a001b3019628c15ea6e33565cc246269b7daf69153c60d850f5262bb5e3345338a3e25b81d82b84281d22f853d3da8">
        : tensor<8x8xi8>
    %w8  = arith.constant dense<"0x929de178022f3933e1c6b75c43c600d46b7a02d5422a5988d756c79c210c3ec9881b609e9fad0b4a9d9166da6cfe17d8214e57187120e561cddeba266e2de015">
        : tensor<8x8xi8>
    %w9  = arith.constant dense<"0x9f0f6a124f3219d2f1c4aa5dcfdb352c166e4d459abad78691d25e2e014933e87ef50fbe97eebf91ad655614bf1a31fbbed75a352576c44532f976feaba1d588">
        : tensor<8x8xi8>
    %w10 = arith.constant dense<"0xacf65a78341b49bc14474b55cc5b7f5df802f04f8db3e07322daf1caa318dbd52da7e3ed073913b4baa47af3ae77105e40e567f0db07b24089b69f24b169183c">
        : tensor<8x8xi8>
  
    %h1  = func.call @matmul(%x,  %w1)  : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h2  = func.call @matmul(%h1, %w2)  : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h3  = func.call @matmul(%h2, %w3)  : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h4  = func.call @matmul(%h3, %w4)  : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h5  = func.call @matmul(%h4, %w5)  : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h6  = func.call @matmul(%h5, %w6)  : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h7  = func.call @matmul(%h6, %w7)  : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h8  = func.call @matmul(%h7, %w8)  : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h9  = func.call @matmul(%h8, %w9)  : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
    %h10 = func.call @matmul(%h9, %w10) : (tensor<8x8xi8>, tensor<8x8xi8>) -> tensor<8x8xi8>
  
    return %h10 : tensor<8x8xi8>
  }
}
