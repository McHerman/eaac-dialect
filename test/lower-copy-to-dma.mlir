// RUN: %eaac-opt %s --split-input-file --eaac-lower-copy-to-dma | FileCheck %s

// Test: memref.copy ops are replaced with eaac.dma_start + eaac.dma_wait pairs.

// CHECK-LABEL: func.func @copy_to_dma
// CHECK-NOT: memref.copy
// CHECK: %[[TOKEN:.*]] = eaac.dma_start(%{{.*}}, %{{.*}}) : memref<128x128xi8>, memref<128x128xi8> -> index
// CHECK-NEXT: eaac.dma_wait(%[[TOKEN]])
// CHECK: %[[TOKEN2:.*]] = eaac.dma_start(%{{.*}}, %{{.*}}) : memref<128x128xi8>, memref<128x128xi8> -> index
// CHECK-NEXT: eaac.dma_wait(%[[TOKEN2]])
module {
  func.func @copy_to_dma(%arg0: memref<128x128xi8>, %arg1: memref<128x128xi8>,
                          %arg2: memref<128x128xi8>) {
    memref.copy %arg0, %arg1 : memref<128x128xi8> to memref<128x128xi8>
    memref.copy %arg1, %arg2 : memref<128x128xi8> to memref<128x128xi8>
    return
  }
}

// -----

// Test: non-copy operations are untouched.

// CHECK-LABEL: func.func @non_copy_untouched
// CHECK: memref.alloc
// CHECK: linalg.fill
// CHECK-NOT: memref.copy
// CHECK: %[[T:.*]] = eaac.dma_start
// CHECK-NEXT: eaac.dma_wait(%[[T]])
module {
  func.func @non_copy_untouched(%arg0: memref<64x64xi8>) -> memref<64x64xi8> {
    %c0_i8 = arith.constant 0 : i8
    %alloc = memref.alloc() : memref<64x64xi8>
    linalg.fill ins(%c0_i8 : i8) outs(%alloc : memref<64x64xi8>)
    memref.copy %alloc, %arg0 : memref<64x64xi8> to memref<64x64xi8>
    return %alloc : memref<64x64xi8>
  }
}
