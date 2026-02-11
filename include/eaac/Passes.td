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

def StaticMemoryAllocation : Pass<"eaac-static-memory-allocation", "mlir::ModuleOp"> {
  let summary = "Allocate memrefs to static scratchpad memory addresses";
  let description = [{
    This pass performs static memory allocation for all memref allocations.
    It uses the minimalloc algorithm to compute optimal memory addresses
    for buffers based on their liveness intervals, minimizing total memory
    usage on the accelerator's scratchpad SRAM.

    The pass:
    1. Collects all memref.alloc operations
    2. Computes liveness intervals for each allocation
    3. Runs the minimalloc algorithm to assign addresses
    4. Replaces allocations with memrefs at fixed addresses

    Example:
    ```mlir
    // Before:
    %buf = memref.alloc() : memref<16x16xf32>
    
    // After:
    %buf = memref.view %base[%c0][] : memref<1048576xi8> to memref<16x16xf32>
    ```
  }];
  let constructor = "mlir::eaac::createStaticMemoryAllocationPass()";
  
  let options = [
    Option<"scratchpadSize", "scratchpad-size", "uint64_t", /*default=*/"1048576",
           "Total size of scratchpad memory in bytes (default: 1MB)">,
    Option<"alignment", "alignment", "uint64_t", /*default=*/"64",
           "Memory alignment requirement in bytes (default: 64)">
  ];
  
  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

#endif // EAAC_PASSES
