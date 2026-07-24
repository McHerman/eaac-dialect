// RUN: %eaac-opt %s --eaac-riscv-kernel-to-llvm | FileCheck %s
// Test: eaac.sem_acquire lowers to a hardware-address wait-loop in isolation.
// No eaac.riscv_execute/riscv_kernel is present, so the other stages of
// --eaac-riscv-kernel-to-llvm (linalg-to-loops, kernel-to-llvm) are no-ops
// on empty kernel/staging lists and only ConvertEAACAcquire fires.
// CHECK-LABEL: func.func @acquire
// CHECK: %[[ADDR:.*]] = llvm.mlir.constant(16388 : i32) : i32
// CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ADDR]] : i32 to !llvm.ptr
// CHECK: llvm.br ^[[LOOP:.*]]
// CHECK: ^[[LOOP]]:
// CHECK:   %[[VAL:.*]] = llvm.load %[[PTR]] : !llvm.ptr -> i32
// CHECK:   %[[DONE:.*]] = llvm.icmp "eq" %[[VAL]], %{{.*}} : i32
// CHECK:   llvm.cond_br %[[DONE]], ^[[EXIT:.*]], ^[[LOOP]]
// CHECK: ^[[EXIT]]:
// CHECK-NOT: eaac.sem_acquire
func.func @acquire(%sem: !eaac.semaphore<addr = 1, gen = 0>,
                   %buf: memref<8x8xi8>,
                   %buf1: memref<8x8xi8>) -> memref<8x8xi8> {
  eaac.sem_acquire %sem, %buf {step_size = 8 : i64} : !eaac.semaphore<addr = 1, gen = 0>, memref<8x8xi8>
  %cst = arith.constant 0 : i8
  %result = memref.alloc() : memref<8x8xi8>
  linalg.fill ins(%cst : i8) outs(%result : memref<8x8xi8>)

  linalg.matmul ins(%buf, %buf1 : memref<8x8xi8>, memref<8x8xi8>)
                outs(%result : memref<8x8xi8>)
  return %result : memref<8x8xi8>
}
