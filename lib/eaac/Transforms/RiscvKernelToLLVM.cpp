//===- RiscvKernelToLLVM.cpp - Lower riscv_execute to LLVM-IR functions --===//
//
// Final lowering pass for the RISC-V execution path. For each eaac.riscv_execute
// op this pass:
//   1. Collects the sem_require / sem_acquire ops in the enclosing eaac.execute
//      to identify input and output semaphores.
//   2. Resolves the hardware addresses and generation tags from the sem_alloc
//      ops that define those semaphores (set by AssignSemaphoreAddresses).
//   3. TODO: Emits a named LLVM-IR function (kernel_sym) that calls
//      eaac_sem_require(addr, gen) / eaac_sem_acquire(addr) and executes the
//      region body lowered via linalg -> loops -> scf -> cf -> llvm.
//   4. TODO: Replaces the riscv_execute with an eaac.riscv_call.
//
// The semaphore runtime stubs are expected to be linked from a pre-verified
// RISC-V AMO library providing:
//   void eaac_sem_require(uint32_t addr, uint32_t gen);  // LR/SC poll loop
//   void eaac_sem_acquire(uint32_t addr);                // AMO store + fence
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "eaac-riscv-kernel-to-llvm"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_RISCVKERNELTOLLVM
#include "eaac/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Resolve the hardware semaphore address and generation tag assigned by
/// AssignSemaphoreAddresses, given a semaphore SSA value.
static std::pair<int64_t, int64_t> resolveSemAttrs(Value sem) {
  auto semAlloc = sem.getDefiningOp<SemAllocOp>();
  if (!semAlloc)
    return {-1, -1};
  auto addrAttr = semAlloc->getAttrOfType<IntegerAttr>("eaac.sem_addr");
  auto genAttr  = semAlloc->getAttrOfType<IntegerAttr>("eaac.sem_gen");
  return {addrAttr ? addrAttr.getInt() : -1,
          genAttr  ? genAttr.getInt()  : -1};
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class RiscvKernelToLLVMPass
    : public impl::RiscvKernelToLLVMBase<RiscvKernelToLLVMPass> {
public:
  using RiscvKernelToLLVMBase::RiscvKernelToLLVMBase;

  void runOnOperation() override {
    llvm::SmallVector<RiscvExecuteOp> kernels;
    getOperation().walk(
        [&](RiscvExecuteOp op) { kernels.push_back(op); });

    for (auto kernel : kernels)
      lowerKernel(kernel);
  }

private:
  void lowerKernel(RiscvExecuteOp riscvExec) {
    auto parentExec = riscvExec->getParentOfType<ExecuteOp>();
    if (!parentExec) {
      riscvExec.emitError("eaac.riscv_execute must be nested inside an "
                          "eaac.execute");
      signalPassFailure();
      return;
    }

    // Collect the synchronization ops from the enclosing execute block.
    // sem_require = inputs the RISC-V core must wait on before starting.
    // sem_acquire = outputs the RISC-V core signals when done.
    llvm::SmallVector<SemRequireOp> inputSems;
    llvm::SmallVector<SemAcquireOp> outputSems;
    for (Operation &op : parentExec->getBody().front()) {
      if (auto req = dyn_cast<SemRequireOp>(&op))
        inputSems.push_back(req);
      else if (auto acq = dyn_cast<SemAcquireOp>(&op))
        outputSems.push_back(acq);
    }

    StringRef sym = riscvExec.getKernelSym();

    LLVM_DEBUG({
      llvm::dbgs() << "[riscv-kernel-to-llvm] kernel: " << sym << "\n";
      for (auto req : inputSems) {
        auto [addr, gen] = resolveSemAttrs(req.getSemaphore());
        llvm::dbgs() << "  require sem_addr=" << addr
                     << " sem_gen=" << gen << "\n";
      }
      for (auto acq : outputSems) {
        auto [addr, gen] = resolveSemAttrs(acq.getSemaphore());
        llvm::dbgs() << "  acquire sem_addr=" << addr << "\n";
      }
    });

    // TODO: Lower riscvExec.getBody() to LLVM-IR via:
    //   - convertLinalgToLoops  (linalg -> scf)
    //   - convertSCFToCF        (scf -> cf)
    //   - convertCFToLLVM + standard arith/memref lowerings (cf -> llvm)
    //
    // Emit as a named LLVM-IR function `sym` with:
    //   - Prologue:  one call to eaac_sem_require(addr, gen) per inputSem
    //   - Body:      the lowered kernel
    //   - Epilogue:  one call to eaac_sem_acquire(addr) per outputSem
    //
    // TODO: Replace this riscv_execute op with an eaac.riscv_call referencing
    // the emitted function by symbol name.
    riscvExec.emitRemark("riscv_execute lowering not yet implemented for: " +
                         sym.str());
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createRiscvKernelToLLVMPass() {
  return std::make_unique<RiscvKernelToLLVMPass>();
}

} // namespace eaac
} // namespace mlir
