// RUN: %eaac-opt %s --eaac-local-staging | FileCheck %s

// Test: All buffers fit in tier 0 (no spilling needed).
// Three 16KB buffers with overlapping lifetimes fit in tier 0's 49152 bytes.

// CHECK-LABEL: func.func @no_spill
// CHECK: memref.alloc() {alignment = 64 : i64, eaac.offset = 0 : i64, eaac.tier = 0 : i64}
// CHECK: memref.alloc() {alignment = 64 : i64, eaac.offset = 16384 : i64, eaac.tier = 0 : i64}
// CHECK: memref.alloc() {alignment = 64 : i64, eaac.offset = 32768 : i64, eaac.tier = 0 : i64}
// CHECK-NOT: memref.copy
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 49152, 147456, 16777216>,
      "num_semaphore_pairs" = 16 : i64,
      "num_semaphore_generations" = 4 : i64
    >
  >
} {
  func.func @no_spill() -> memref<128x128xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<128x128xi8>
    %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<128x128xi8>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<128x128xi8>
    linalg.fill ins(%c0_i8 : i8) outs(%alloc_0 : memref<128x128xi8>)
    linalg.fill ins(%c1_i8 : i8) outs(%alloc_1 : memref<128x128xi8>)
    linalg.add ins(%alloc_0, %alloc_1 : memref<128x128xi8>, memref<128x128xi8>) outs(%alloc_2 : memref<128x128xi8>)
    return %alloc_2 : memref<128x128xi8>
  }
}
