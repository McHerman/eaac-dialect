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

def LowerCopyToDma : Pass<"eaac-lower-copy-to-dma", "mlir::ModuleOp"> {
  let summary = "Lower memref.copy to async DMA start/wait pairs";
  let description = [{
    This pass replaces every `memref.copy` operation with an
    `eaac.dma_start` / `eaac.dma_wait` pair, producing 1:1 blocking
    semantics. Future passes can hoist `dma_wait` to overlap compute
    and data movement.

    The pass:
    1. Walks all memref.copy operations
    2. Replaces each with eaac.dma_start(%src, %dst) -> %token
    3. Inserts eaac.dma_wait(%token) immediately after
    4. Erases the original memref.copy
  }];
  let constructor = "mlir::eaac::createLowerCopyToDmaPass()";

  let dependentDialects = [
    "mlir::memref::MemRefDialect",
    "mlir::eaac::EAACDialect"
  ];
}

def EncodeDependencies : Pass<"eaac-encode-dependencies", "mlir::ModuleOp"> {
  let summary = "Wrap DMA and compute ops in async.execute regions";
  let description = [{
    Wraps every eaac.dma_start and linalg op in an async.execute region.
    Each wrapped op produces an async.token. Dependency tokens are not
    threaded in this pass — a later pass analyzes data dependencies and
    connects the tokens.
  }];
  let constructor = "mlir::eaac::createEncodeDependenciesPass()";

  let dependentDialects = [
    "mlir::async::AsyncDialect",
    "mlir::eaac::EAACDialect"
  ];
}

def FindAsyncDependency : Pass<"eaac-find-async-dependency", "mlir::ModuleOp"> {
  let summary = "Find and wire async.token dependencies between async.execute regions";
  let description = [{
    For each async.execute region, inspects the memref operands of its inner
    ops and finds producers (other async.execute regions that write to those
    memrefs) via SSA use-def chains. Adds the producer tokens as dependencies
    on the consumer async.execute.
  }];
  let constructor = "mlir::eaac::createFindAsyncDependencyPass()";

  let dependentDialects = [
    "mlir::async::AsyncDialect",
    "mlir::eaac::EAACDialect"
  ];
}

def LowerAsyncToSemaphore : Pass<"eaac-lower-async-to-semaphore", "mlir::ModuleOp"> {
  let summary = "Lower async.execute regions and tokens to hardware semaphores";
  let description = [{
    Converts async.token dependencies between async.execute regions into
    hardware semaphore operations. Each token dependency becomes a sem_alloc,
    producer ops get semaphore args for signaling, consumer ops get semaphore
    args for waiting. DMA ops keep their semaphore args directly, linalg ops
    are wrapped in eaac.compute. Async.execute regions are stripped.
  }];
  let constructor = "mlir::eaac::createLowerAsyncToSemaphorePass()";

  let dependentDialects = [
    "mlir::async::AsyncDialect",
    "mlir::arith::ArithDialect",
    "mlir::eaac::EAACDialect"
  ];
}

#endif // EAAC_PASSES
