//===- LegalizeForEaacHw.cpp - Legalize ops for EAAC hardware -------------===//
//
// Dialect conversion pass that legalizes an MLIR module for the EAAC hardware
// target. Ops supported by native hardware units (eaac.matmul, eaac.dma_start,
// etc.) and infrastructure ops (memref moves, arith constants) are legal.
// Any other op found inside an eaac.execute body is illegal and gets wrapped
// in an eaac.riscv_execute region for later lowering to the RISC-V core.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "eaac-legalize-for-hw"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_LEGALIZEFOREAACHW
#include "eaac/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Conversion pattern
//===----------------------------------------------------------------------===//

/// Wraps any hardware-unsupported op in an eaac.riscv_execute region.
///
/// Fires on any op the ConversionTarget marks illegal — meaning any op inside
/// an eaac.execute body whose dialect is not explicitly legal. The op's memref
/// operands (produced by surrounding sem_require / sem_acquire ops) are
/// captured from the enclosing scope since RiscvExecuteOp is not isolated.
struct UnsupportedOpToRiscvExecute : public RewritePattern {
  UnsupportedOpToRiscvExecute(MLIRContext *ctx, uint64_t *counter)
      : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx),
        counter(counter) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Already inside a riscv_execute — don't recurse.
    if (op->getParentOfType<RiscvExecuteOp>())
      return failure();
    // Only wrap direct children of eaac.execute. Ops in nested regions
    // (e.g. linalg.yield inside linalg.generic) are cloned along with their
    // parent and must not be processed separately.
    Operation *parent = op->getParentOp();
    if (!parent || !isa<ExecuteOp>(parent))
      return failure();

    auto executeOp = cast<ExecuteOp>(parent);
    std::string sym = "riscv_kernel_" + std::to_string((*counter)++);

    // Replace the whole eaac.execute with an eaac.riscv_execute, moving its
    // entire body over as-is. One execute region maps to one execution unit,
    // so a single unsupported op is enough to push the whole region to the
    // RISC-V core.
    rewriter.setInsertionPoint(executeOp);
    auto riscvExec = RiscvExecuteOp::create(
        rewriter, executeOp.getLoc(),
        StringAttr::get(executeOp->getContext(), sym));
    rewriter.inlineRegionBefore(executeOp.getBody(), riscvExec.getBody(),
                                riscvExec.getBody().end());
    rewriter.eraseOp(executeOp);

    LLVM_DEBUG(llvm::dbgs() << "[legalize-for-hw] " << executeOp->getName()
                             << " -> " << sym << "\n");
    return success();
  }

  uint64_t *counter;
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class LegalizeForEaacHwPass
    : public impl::LegalizeForEaacHwBase<LegalizeForEaacHwPass> {
public:
  using LegalizeForEaacHwBase::LegalizeForEaacHwBase;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    ConversionTarget target(*ctx);

    // Ops from these dialects are always legal — they represent hardware-native
    // operations or infrastructure that all execution units share.
    target.addLegalDialect<EAACDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();


    // Illegal operations have to be wrapped in risc-v execution region

    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      if (op->getParentOfType<RiscvExecuteOp>())
        return true;
      // Nested-region ops (e.g. linalg.yield inside linalg.generic) are legal:
      // they get cloned with their parent, not converted independently.
      Operation *parent = op->getParentOp();
      if (!parent || !isa<ExecuteOp>(parent))
        return true;
      return false;
    });

    uint64_t counter = 0;
    RewritePatternSet patterns(ctx);
    patterns.add<UnsupportedOpToRiscvExecute>(ctx, &counter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createLegalizeForEaacHwPass() {
  return std::make_unique<LegalizeForEaacHwPass>();
}

} // namespace eaac
} // namespace mlir
