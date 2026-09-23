//===- Passes.h - EAAC dialect passes ---------------------------*- C++ -*-===//
//
// Header file for EAAC dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_PASSES_H
#define EAAC_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace eaac {

//===----------------------------------------------------------------------===//
// Pass Declarations (Generated)
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "eaac/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

/// Creates a pass to collect alloc/dealloc operations and print liveness info.
std::unique_ptr<Pass> createCollectAllocDeallocPass();

/// Creates pass to statically allocate memory
std::unique_ptr<Pass> createMemoryAllocPass();

/// Creates pass to insert local SRAM staging for compute operations
std::unique_ptr<Pass> createLocalStagingPass();

/// Creates pass to lower memref.copy to eaac.dma_start/dma_wait pairs
std::unique_ptr<Pass> createLowerCopyToDmaPass();

/// Creates pass to wrap DMA and compute ops in async.execute regions
std::unique_ptr<Pass> createEncodeDependenciesPass();

/// Creates pass to find and wire async.token dependencies between async.execute regions
std::unique_ptr<Pass> createFindAsyncDependencyPass();

/// Creates pass that adds async.token edges for allocator-induced aliases
std::unique_ptr<Pass> createFindAliasDependencyPass();

/// Creates pass that corrects broadcast operations
std::unique_ptr<Pass> createCorrectBroadcastPass();

/// Creates pass to insert eaac.require ops pairing tokens with memrefs
std::unique_ptr<Pass> createInsertRequirePass();

/// Creates pass to lower async.execute/tokens to hardware semaphores
std::unique_ptr<Pass> createLowerAsyncToSemaphorePass();

/// Creates pass to replace function args/returns with eaac.load/store
std::unique_ptr<Pass> createInsertLoadStorePass();

/// Creates pass to replace function args/returns with eaac.load/store
std::unique_ptr<Pass> createSchedulePass();

/// Creates pass to strip schedule annotations
std::unique_ptr<Pass> createStripSchedulePass();

// Creates pass to optimize semaphore allocations
std::unique_ptr<Pass> createSemOptimizePass();

// Creates pass to optimize semaphore allocations using queue-based scheduling
std::unique_ptr<Pass> createSemOptimizeQueuePass();

/// Creates pass to assign static hardware addresses to semaphore pairs
std::unique_ptr<Pass> createAssignSemaphoreAddressesPass();

/// Creates pass to wrap hardware-unsupported ops in eaac.riscv_execute regions
std::unique_ptr<Pass> createLegalizeForEaacHwPass();

/// Creates pass to outline eaac.riscv_execute regions into func.func ops
std::unique_ptr<Pass> createRiscvKernelToFunctionPass();

/// Creates pass to lower eaac and memref ops to LLVM dialect
std::unique_ptr<Pass> createLowerEaacMemrefToLLVMPass();

/// Creates pass to lower eaac.riscv_kernel func.func ops to LLVM dialect
std::unique_ptr<Pass> createRiscvKernelToLLVMPass();

/// Creates pass to split LLVM dialect content from EAAC dialect content
std::unique_ptr<Pass> createSplitLLVMFromEAACPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "eaac/Passes.h.inc"

} // namespace eaac
} // namespace mlir

#endif // EAAC_PASSES_H
