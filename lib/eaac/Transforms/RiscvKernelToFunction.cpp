#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "eaac-riscv-kernel-to-function"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_RISCVKERNELTOFUNCTION
#include "eaac/Passes.h.inc"

namespace {

struct CreateRiscVFunctions : OpRewritePattern<eaac::RiscvExecuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(eaac::RiscvExecuteOp op,
                                PatternRewriter &rewriter) const override {

    auto module = op->getParentOfType<ModuleOp>();
    OpBuilder::InsertionGuard guard(
        rewriter); // saves + restores insertion point
    rewriter.setInsertionPointToEnd(module.getBody());

    StringRef sym = op.getKernelSym();

    llvm::SetVector<Value> captured;
    op.getBody().walk([&](Operation *inner) {
      for (Value operand : inner->getOperands())
        if (!op.getBody().isAncestor(operand.getParentRegion()))
          captured.insert(operand);
    });

    llvm::SmallVector<Type> argTypes;
    for (Value v : captured)
      argTypes.push_back(v.getType());

    auto func =
        func::FuncOp::create(rewriter, op.getLoc(), sym,
                             FunctionType::get(op->getContext(), argTypes, {}));

    func->setAttr("eaac.riscv_kernel", UnitAttr::get(op->getContext()));

    // Map captured values to the new function's block arguments.
    Block *body = rewriter.createBlock(
        &func.getBody(), {}, argTypes,
        SmallVector<Location>(argTypes.size(), op.getLoc()));
    IRMapping mapping;
    for (auto [val, arg] : llvm::zip(captured, body->getArguments()))
      mapping.map(val, arg);

    // Clone the ops from inside the riscv_execute body into the function.
    rewriter.setInsertionPointToStart(body);
    for (Operation &inner : op.getBody().front())
      rewriter.clone(inner, mapping);

    rewriter.setInsertionPointToEnd(body);
    func::ReturnOp::create(rewriter, func.getLoc());

    rewriter.setInsertionPoint(op);
    // Replace the riscv_execute with a call to the outlined function.
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, func, SmallVector<Value>(captured.begin(), captured.end()));

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class RiscvKernelToFunctionPass
    : public impl::RiscvKernelToFunctionBase<RiscvKernelToFunctionPass> {
public:
  using RiscvKernelToFunctionBase::RiscvKernelToFunctionBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CreateRiscVFunctions>(&getContext());

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createRiscvKernelToFunctionPass() {
  return std::make_unique<RiscvKernelToFunctionPass>();
}

} // namespace eaac
} // namespace mlir
