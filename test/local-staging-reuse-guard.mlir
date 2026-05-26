// RUN: %eaac-opt %s --split-input-file --eaac-local-staging | FileCheck %s

// Two non-overlapping buffers %a and %b can normally share the same offset
// after the allocator reuses the address. With reuse_guard left out (default
// 0), the second allocation is packed on top of the first.

// CHECK-LABEL: func.func @no_guard
// CHECK: %[[R:.*]] = memref.alloc() {alignment = 64 : i64, eaac.offset = 0 : i64, eaac.tier = 0 : i64}
// CHECK: %[[A:.*]] = memref.alloc() {alignment = 64 : i64, eaac.offset = 16384 : i64, eaac.tier = 0 : i64}
// CHECK: %[[B:.*]] = memref.alloc() {alignment = 64 : i64, eaac.offset = 16384 : i64, eaac.tier = 0 : i64}
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 49152, 147456, 16777216>,
      "num_semaphore_pairs" = 16 : i64
    >
  >
} {
  func.func @no_guard() -> memref<128x128xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %r = memref.alloc() {alignment = 64 : i64} : memref<128x128xi8>
    %a = memref.alloc() {alignment = 64 : i64} : memref<128x128xi8>
    linalg.fill ins(%c0_i8 : i8) outs(%a : memref<128x128xi8>)
    linalg.copy ins(%a : memref<128x128xi8>) outs(%r : memref<128x128xi8>)
    %b = memref.alloc() {alignment = 64 : i64} : memref<128x128xi8>
    linalg.fill ins(%c1_i8 : i8) outs(%b : memref<128x128xi8>)
    linalg.add ins(%r, %b : memref<128x128xi8>, memref<128x128xi8>) outs(%r : memref<128x128xi8>)
    return %r : memref<128x128xi8>
  }
}

// -----

// With reuse_guard=4, the allocator extends each buffer's lifespan upper
// bound by 4 time units when handed to the solver. %a's padded interval now
// overlaps %b's start, so the solver must place %b at a different offset.

// CHECK-LABEL: func.func @with_guard
// CHECK: %[[R:.*]] = memref.alloc() {alignment = 64 : i64, eaac.offset = 0 : i64, eaac.tier = 0 : i64}
// CHECK: %[[A:.*]] = memref.alloc() {alignment = 64 : i64, eaac.offset = 16384 : i64, eaac.tier = 0 : i64}
// CHECK: %[[B:.*]] = memref.alloc() {alignment = 64 : i64, eaac.offset = 32768 : i64, eaac.tier = 0 : i64}
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "EAAC" = #dlti.target_device_spec<
      "tier_capacities" = array<i64: 49152, 147456, 16777216>,
      "reuse_guard" = 4 : i64,
      "num_semaphore_pairs" = 16 : i64
    >
  >
} {
  func.func @with_guard() -> memref<128x128xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c1_i8 = arith.constant 1 : i8
    %r = memref.alloc() {alignment = 64 : i64} : memref<128x128xi8>
    %a = memref.alloc() {alignment = 64 : i64} : memref<128x128xi8>
    linalg.fill ins(%c0_i8 : i8) outs(%a : memref<128x128xi8>)
    linalg.copy ins(%a : memref<128x128xi8>) outs(%r : memref<128x128xi8>)
    %b = memref.alloc() {alignment = 64 : i64} : memref<128x128xi8>
    linalg.fill ins(%c1_i8 : i8) outs(%b : memref<128x128xi8>)
    linalg.add ins(%r, %b : memref<128x128xi8>, memref<128x128xi8>) outs(%r : memref<128x128xi8>)
    return %r : memref<128x128xi8>
  }
}
