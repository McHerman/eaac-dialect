module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 8096, 16384, 32768>,
      "alias_check_length" = 128 : i64,
      "num_semaphore_pairs" = 16 : i64,
      "num_semaphore_generations" = 4 : i64
    >
  >
} {
  func.func @main(%arg0: tensor<8x8xi8>, %arg1: tensor<8x8xi8>) -> tensor<1x8xi8> {
    %a = arith.constant dense<"0xe65c61df33bd6a4b">
        : tensor<1x8xi8>
    %b = arith.constant dense<"0xe32c6219e76b12a417bec435022020261586c585b47df08e8183cc78d76954386b759a551d7dc470a55bdbb60189b3803f940cae3b13817e941973ae20c4931e">
        : tensor<8x8xi8>

    %1 = arith.constant 0 : i8
    %42 = tensor.empty() : tensor<1x8xi8>
    %43 = linalg.fill ins(%1 : i8) outs(%42 : tensor<1x8xi8>) -> tensor<1x8xi8>
    %44 = linalg.matmul ins(%a, %arg0 : tensor<1x8xi8>, tensor<8x8xi8>) outs(%43 : tensor<1x8xi8>) -> tensor<1x8xi8>

    %13 = tensor.empty() : tensor<1x8xi8>
    %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%44 : tensor<1x8xi8>) outs(%13 : tensor<1x8xi8>) {
    ^bb1(%15: i8, %16: i8):
      %17 = arith.constant 0 : i8
      %18 = arith.cmpi sgt, %15, %17 : i8
      %19 = arith.select %18, %15, %17 : i8
      linalg.yield %19 : i8
    } -> tensor<1x8xi8>

    return %14 : tensor<1x8xi8>
  }
}
