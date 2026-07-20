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

    Memory hierarchy levels (encoded as tiers):
      - 0: Local SRAM (per-unit scratchpad, required for arithmetic ops)
      - 1: Global SRAM (shared on-chip memory)
      - 2: DRAM (external memory)

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

def FindAliasDependency : Pass<"eaac-find-alias-dependency", "mlir::ModuleOp"> {
  let summary = "Add async.token edges for allocator-induced memref aliases";
  let description = [{
    Runs after LocalStaging has annotated memref allocs with eaac.offset and
    eaac.tier, and after FindAsyncDependency has wired SSA-level token edges.

    LocalStaging's reuse_guard knob spreads addresses to avoid aliasing, but
    when memory pressure forces address reuse, two SSA-distinct memrefs can
    share the same physical {tier, offset} range. The single-writer invariant
    that FindAsyncDependency relies on is broken at the physical level, so the
    consumer of the new buffer can race against the last reader of the old
    buffer.

    This pass detects those alias pairs and adds async.token edges between the
    relevant async.execute regions so that downstream lowering (InsertRequire,
    LowerAsyncToSemaphore) materializes a hardware semaphore chain.

    Edges that are already implied by transitive ordering in the existing
    token graph are skipped. The DLTI key "alias_check_length" bounds the
    op-time distance at which alias pairs are still considered; pairs whose
    time gap exceeds the window are ignored (and will fall back to a fence
    once fence insertion lands).
  }];
  let constructor = "mlir::eaac::createFindAliasDependencyPass()";

  let dependentDialects = [
    "mlir::async::AsyncDialect",
    "mlir::eaac::EAACDialect"
  ];
}

def CorrectBroadcast : Pass<"eaac-correct-broadcast", "mlir::ModuleOp"> {
  let summary = "TODO: one-line summary";
  let description = [{
    TODO: longer description.
  }];
  let constructor = "mlir::eaac::createCorrectBroadcastPass()";

  let dependentDialects = [
    "mlir::async::AsyncDialect",
    "mlir::eaac::EAACDialect"
  ];
}

def InsertRequire : Pass<"eaac-insert-require", "mlir::ModuleOp"> {
  let summary = "Insert eaac.require ops to pair async tokens with memrefs";
  let description = [{
    For each async.execute region, inspects the token dependencies and
    traces them back to the producer ops to determine which memrefs they
    guard. Inserts eaac.require ops that explicitly pair each token with
    its corresponding memref, and replaces uses of the original memref
    inside the consumer region with the require result.
  }];
  let constructor = "mlir::eaac::createInsertRequirePass()";

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

def InsertLoadStore : Pass<"eaac-insert-load-store", "mlir::ModuleOp"> {
  let summary = "Flatten hierarchy, replace main function parameters with load, returns with store";
  let description = [{

  }];
  let constructor = "mlir::eaac::createInsertLoadStorePass()";

  let dependentDialects = [
    "mlir::eaac::EAACDialect"
  ];
}


def AssignSemaphoreAddresses : Pass<"eaac-assign-semaphore-addresses", "mlir::ModuleOp"> {
  let summary = "Assign static hardware addresses to semaphore pairs";
  let description = [{
    Uses a linear scan allocator to assign static hardware addresses to
    semaphore pairs created by the LowerAsyncToSemaphore pass. Semaphores
    are self-sequencing, so no spilling is required.

    If the number of live semaphores exceeds the hardware limit, the pass
    reuses the address of the soonest-finishing semaphore, provided its
    consumer has already started executing. If this safety condition cannot
    be met, the pass emits an error and aborts.

    Annotates each eaac.sem_alloc with:
      - eaac.sem_addr: the assigned hardware address (index)
      - eaac.sem_gen: per-address generation tag (wraps modulo the DLTI
        entry `num_semaphore_generations`); successive allocations on the
        same hardware address get distinct tags so the fabric can reject
        stale signals from prior generations
      - eaac.sem_reused: unit attr if the address was reused from a live sem
  }];
  let constructor = "mlir::eaac::createAssignSemaphoreAddressesPass()";

  let dependentDialects = [
    "mlir::eaac::EAACDialect"
  ];
}

def LegalizeForEaacHw : Pass<"eaac-legalize-for-hw", "mlir::ModuleOp"> {
  let summary = "Wrap hardware-unsupported ops in eaac.riscv_execute regions";
  let description = [{
    Runs after LinalgToEaac. Any linalg op still present inside an eaac.execute
    body could not be lowered to a native hardware unit and must instead run on
    the embedded RISC-V core.

    For each such op this pass:
      1. Creates an eaac.riscv_execute region at the op's location with a
         unique kernel symbol name derived from the enclosing function.
      2. Moves the op into the riscv_execute body.

    The surrounding sem_require / sem_acquire ops in the enclosing execute block
    remain in place — they already encode the synchronization contract and will
    be consumed by RiscvKernelToLLVM during final lowering.
  }];
  let constructor = "mlir::eaac::createLegalizeForEaacHwPass()";

  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::eaac::EAACDialect"
  ];
}

def RiscvKernelToFunction : Pass<"eaac-riscv-kernel-to-function", "mlir::ModuleOp"> {
  let summary = "Outline eaac.riscv_execute regions into standalone func.func ops";
  let description = [{
    Runs after LegalizeForEaacHw and AssignSemaphoreAddresses.

    For each eaac.riscv_execute op this pass:
      1. Collects the values captured from the enclosing eaac.execute scope
         (memrefs produced by sem_require ops) — these become function arguments.
      2. Emits a func.func named by kernel_sym whose body is the region body
         with all captured values replaced by block arguments.
      3. TODO: inserts eaac_sem_require / eaac_sem_acquire calls as prologue and
         epilogue in the emitted function using the hardware addresses resolved
         by AssignSemaphoreAddresses.
      4. TODO: replaces the eaac.riscv_execute with an eaac.riscv_call referencing
         the emitted function by symbol.

    After this pass the emitted func.func ops contain plain linalg/arith/memref
    ops and are ready for the standard linalg → loops → cf → llvm lowering chain.
  }];
  let constructor = "mlir::eaac::createRiscvKernelToFunctionPass()";

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::eaac::EAACDialect"
  ];
}

def LowerEaacMemrefToLLVM : Pass<"eaac-lower-memref-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower eaac and memref ops to LLVM dialect";
  let description = [{
    TODO: longer description.
  }];
  let constructor = "mlir::eaac::createLowerEaacMemrefToLLVMPass()";

  let dependentDialects = [
    "mlir::eaac::EAACDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def RiscvKernelToLLVM : Pass<"eaac-riscv-kernel-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower eaac.riscv_kernel func.func ops to LLVM dialect";
  let description = [{
    Selects func.func ops marked with the eaac.riscv_kernel attribute and
    applies the standard lowering chain (linalg → loops → cf → llvm) using
    conversion patterns directly, leaving the surrounding EAAC IR untouched.
  }];
  let constructor = "mlir::eaac::createRiscvKernelToLLVMPass()";

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def SplitLLVMFromEAAC : Pass<"eaac-split-llvm-from-eaac", "mlir::ModuleOp"> {
  let summary = "Extract RISC-V staging kernels into a standalone LLVM IR file";
  let description = [{
    Moves constant parameters into the body of MLIR wrapper functions, removes
    memrefs. Converts wrapper functions into llvm. Gathers all llvm functions
    and into seperate module, converts to llvmir and dumps to file.
  }];
  let constructor = "mlir::eaac::createSplitLLVMFromEAACPass()";

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::LLVM::LLVMDialect",
    "mlir::eaac::EAACDialect"
  ];

  let options = [
    Option<"llvmOutputFile", "llvm-output-file", "std::string", /*default=*/"\"\"",
           "Path to write the extracted LLVM IR module (.ll) to; if empty, "
           "the extracted kernels are discarded">
  ];
}

#endif // EAAC_PASSES
