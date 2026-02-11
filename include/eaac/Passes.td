//===- Passes.td - EAAC dialect passes ---------------------*- tablegen -*-===//
//
// Defines passes for the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_PASSES
#define EAAC_PASSES

include "mlir/Pass/PassBase.td"

def CollectAllocDealloc : Pass<"eaac-collect-alloc-dealloc", "mlir::ModuleOp"> {
  let summary = "Collect memref alloc/dealloc operations and print liveness info";
  let description = [{
    This pass walks across all memref.alloc and memref.dealloc operations,
    collects them in separate lists, and prints liveness information.

    This is an experimental pass for understanding memory allocation patterns
    before performing static memory allocation.

    The pass:
    1. Collects all memref.alloc operations
    2. Collects all memref.dealloc operations
    3. Computes operation numbering for liveness
    4. Prints detailed liveness information for each allocation
  }];
  let constructor = "mlir::eaac::createCollectAllocDeallocPass()";

  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

#endif // EAAC_PASSES
