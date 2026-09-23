module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 32768, 65536, 65536>,
      "alias_check_length" = 4096 : i64,
      "reuse_guard" = 100000 : i64,
      "num_semaphore_pairs" = 64 : i64,
      "num_semaphore_generations" = 4 : i64,
      "bus_size" = 64 : i64,
      "riscv_sem_base" = 32768 : i64
    >
  >
} {
  eaac.schedule @schedule {
    eaac.hw_alloc @unit0 : !eaac.gemm<2 32>
    eaac.hw_alloc @unit1 : !eaac.risc<1 32>
  }

  func.func @main(%0: tensor<16x16xi8>, %1: tensor<16x16xi8>, %2: tensor<16x16xi8>, %3: tensor<16x16xi8>, %4: tensor<16x16xi8>, %5: tensor<16x16xi8>, %6: tensor<16x16xi8>, %7: tensor<16x16xi8>) -> tensor<16x16xi8> {
    %c0_i32 = arith.constant 0 : i32
    %w0 = arith.constant dense<"0x390C8C7D7247342CD8100F2F6F770D65D670E58E0351D8AE8E4F6EAC342FC231B7B08716EB3FC12896B96223177494287733C28EE8BA53BDB56B8824577D53ECC28A70A61C7510A1CD89216CA16CFFCAEA4987477E86DBCCB97046FC2E18384E51D820C5C3EF80053A88AE3996DE50E801865B3698654EBF5200A5FA0939B99D7A1D7B282BF8234041F35487D86C669FCCBFE0E73D7E7320AD0A757003241E752210A924798EF86D43F27CF2D0613031DCB5D8D2EF1B321FCEAD377F6261E547D85D8EEC7F26E23219072F7955D0F8F66DCD1E54C201C787E892D8F94F61976F1D1FA01D19F4501D295F232278CE3D7E1429D6A18568A07A87CA4399EAA12504"> : tensor<16x16xi8>
    %acc0_0 = tensor.empty() : tensor<16x16xi32>
    %acc0_7 = tensor.empty() : tensor<16x16xi32>

    %acc_7 = linalg.fill ins(%c0_i32 : i32) outs(%acc0_7 : tensor<16x16xi32>) -> tensor<16x16xi32>
    %m7 = linalg.matmul ins(%7, %w0 : tensor<16x16xi8>, tensor<16x16xi8>) outs(%acc_7 : tensor<16x16xi32>) -> tensor<16x16xi32>

    %bias = arith.constant dense<"0xB8FEEE32F0A368BDA0D317714A0885D5974E64A875C27DFFAC83FAFBEB56B45647FA5E1E11261803D34676224D046FE9BF1EF7F90803D206088C9208DC5B3631"> : tensor<16xi32>
    %biased_out = tensor.empty() : tensor<16x16xi32>
    %biased = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%acc_7, %bias : tensor<16x16xi32>, tensor<16xi32>) outs(%biased_out : tensor<16x16xi32>) {
    ^bb0(%a: i32, %b: i32, %o: i32):
      %s = arith.addi %a, %b : i32
      linalg.yield %s : i32
    } -> tensor<16x16xi32>
    %relu_out = tensor.empty() : tensor<16x16xi32>
    %relu = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%biased : tensor<16x16xi32>) outs(%relu_out : tensor<16x16xi32>) {
    ^bb0(%x: i32, %o: i32):
      %rc0 = arith.constant 0 : i32
      %cond = arith.cmpi sgt, %x, %rc0 : i32
      %r = arith.select %cond, %x, %rc0 : i32
      linalg.yield %r : i32
    } -> tensor<16x16xi32>
    %quant_out = tensor.empty() : tensor<16x16xi8>
    %quant = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%relu : tensor<16x16xi32>) outs(%quant_out : tensor<16x16xi8>) {
    ^bb0(%in: i32, %out: i8):
      %M0 = arith.constant 1091133696 : i64
      %rsh = arith.constant 35 : i64
      %nudge = arith.constant 17179869184 : i64
      %zp = arith.constant -128 : i64
      %lo = arith.constant -128 : i64
      %hi = arith.constant 127 : i64
      %x64 = arith.extsi %in : i32 to i64
      %prod = arith.muli %x64, %M0 : i64
      %rndd = arith.addi %prod, %nudge : i64
      %shft = arith.shrsi %rndd, %rsh : i64
      %bzp = arith.addi %shft, %zp : i64
      %locnd = arith.cmpi sgt, %bzp, %lo : i64
      %clo = arith.select %locnd, %bzp, %lo : i64
      %hicnd = arith.cmpi slt, %clo, %hi : i64
      %chi = arith.select %hicnd, %clo, %hi : i64
      %trunc = arith.trunci %chi : i64 to i8
      linalg.yield %trunc : i8
    } -> tensor<16x16xi8>
    return %quant : tensor<16x16xi8>
  }
}
