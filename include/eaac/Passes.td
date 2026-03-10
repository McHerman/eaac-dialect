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

def MemoryAlloc : Pass<"eaac-memory-alloc", "mlir::ModuleOp"> {
  let summary = "Basic static memory allocation pass";
  let description = [{
    This pass walks across all memref.alloc and memref.dealloc operations,
    collects them in separate lists.

    The pass then uses this information to create a minimalloc problem which
    is then solved.

    The pass:
    1. Collects all memref.alloc operations
    2. Collects all memref.dealloc operations
    3. Computes operation numbering for liveness
    4. Creates minimalloc problem
    5. Solves said minimalloc problem
  }];
  let constructor = "mlir::eaac::createMemoryAllocPass()";

  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

def LocalStaging : Pass<"eaac-local-staging", "mlir::ModuleOp"> {
  let summary = "Insert local SRAM staging for compute operations";
  let description = [{
    This pass inserts memref.copy operations to stage data into local SRAM
    before compute operations (like linalg ops) and copy results back afterward.

    Memory hierarchy levels (encoded as memory spaces):
      - 1: DRAM (external memory)
      - 2: Global SRAM (shared on-chip memory)  
      - 3: Local SRAM (per-unit scratchpad, required for arithmetic ops)

    The pass:
    1. Walks through memrefs (linalg ops)
    2. For each memref operand not already in local SRAM (memory_space=3):
       - Allocates a local buffer with the same shape
       - Inserts memref.copy to stage data in before the operation
    3. For output operands:
       - Inserts memref.copy to write results back after the operation
    4. Rewrites the compute op to use local buffers
    5. Deallocates local buffers after use
  }];
  let constructor = "mlir::eaac::createLocalStagingPass()";

  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

#endif // EAAC_PASSES
