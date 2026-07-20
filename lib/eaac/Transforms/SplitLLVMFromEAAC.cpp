#include "eaac/Passes.h"

#include "eaac/Dialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/BitVector.h"
#include <cstddef>

#define DEBUG_TYPE "eaac-split-llvm-from-eaac"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_SPLITLLVMFROMEAAC
#include "eaac/Passes.h.inc"

namespace {

// Fires inside RISC-V staging wrapper functions only.
//
// RiscvKernelToLLVM leaves an unrealized_conversion_cast from each
// memref<..., #eaac.mem<tier, offset>> argument to the corresponding
// !llvm.struct because the staging func.func stays alive (it is legal in the
// target). This pattern replaces that cast with a manually-built memref
// descriptor whose alignedPtr field is inttoptr(offset), so the RISC-V core
// can reach the data at the hardware-assigned address.
struct CopyConstants : OpRewritePattern<func::FuncOp> {

  using OpRewritePattern::OpRewritePattern;


  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {

    if (!op->hasAttr("eaac.riscv_staging_kernel"))
      return failure();

    // Identify the arg index of every non-memref (i.e. scalar constant) arg.
    llvm::SmallVector<int64_t> constantIdx;

    for (auto [idx, argType] : llvm::enumerate(op.getFunctionType().getInputs())) {
      if (isa<MemRefType>(argType))
        continue;

      constantIdx.push_back(static_cast<int64_t>(idx));
    }


    LLVM_DEBUG(llvm::dbgs() << "FOUND ARGS");

    // Find call sites via the symbol table (func.call references the callee
    // by name, not by SSA value, so there is no use-list on the symbol name).
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto uses = SymbolTable::getSymbolUses(op, moduleOp);                                      

    llvm::SmallVector<std::pair<int64_t, arith::ConstantOp>> constantOps;

    // Identify constants.
    if(uses) {
      for (SymbolTable::SymbolUse use : *uses) {
        auto call = dyn_cast<func::CallOp>(use.getUser());
        if (!call)
          continue;


        for (auto index : constantIdx) {
          auto defOp = call.getOperand(index).getDefiningOp();
          if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(defOp))
            constantOps.push_back({index, constOp});
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "FOUND OPS");


    rewriter.setInsertionPointToStart(&op.getBody().front());

    // Copy ops to func and replace the corresponding argument's uses with them.
    for (auto [index, constant] : constantOps) {
      Operation *copy = rewriter.clone(*constant);
      rewriter.replaceAllUsesWith(op.getArgument(index), copy->getResult(0));
    }


    LLVM_DEBUG(llvm::dbgs() << "INSERTED CONST");


    llvm::BitVector argsToErase(op.getNumArguments());
    for (auto [index, constant] : constantOps)
      argsToErase.set(index);

    if (uses) {
      for (SymbolTable::SymbolUse use : *uses) {
        auto call = dyn_cast<func::CallOp>(use.getUser());
        if (!call)
          continue;

        rewriter.modifyOpInPlace(call, [&]() {
          call->eraseOperands(argsToErase);
        });
      }
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.eraseArguments(argsToErase);
    });


    MLIRContext *ctx = op->getContext();

    rewriter.modifyOpInPlace(op, [&]() {
      //op.setSymName(implName);
      op->removeAttr("eaac.riscv_staging_kernel");
      op->setAttr("eaac.riscv_staging_kernel_no_const", UnitAttr::get(ctx));
    });

    return success();
  }

};


struct RemoveMemrefArgs : OpRewritePattern<func::FuncOp> {

  using OpRewritePattern::OpRewritePattern;


  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {

    if (!op->hasAttr("eaac.riscv_staging_kernel_no_const"))
      return failure();

    llvm::SmallVector<int64_t> memrefIdx;

    for (auto [idx, argType] : llvm::enumerate(op.getFunctionType().getInputs())) {
      if (!isa<MemRefType>(argType))
        continue;

      if (!op.getArgument(idx).use_empty())
        continue;

      memrefIdx.push_back(static_cast<int64_t>(idx));
    }


    if (!op->hasAttr("eaac.riscv_staging_kernel_no_const"))
      return failure();


    llvm::BitVector argsToErase(op.getNumArguments());
    for (auto index : memrefIdx)
      argsToErase.set(index);


    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto uses = SymbolTable::getSymbolUses(op, moduleOp);                                      

    if (uses) {
      for (SymbolTable::SymbolUse use : *uses) {
        auto call = dyn_cast<func::CallOp>(use.getUser());
        if (!call)
          continue;

        rewriter.modifyOpInPlace(call, [&]() {
          call->eraseOperands(argsToErase);
        });
      }
    }

    rewriter.modifyOpInPlace(op, [&]() {
      op.eraseArguments(argsToErase);
    });


    MLIRContext *ctx = op->getContext();

    rewriter.modifyOpInPlace(op, [&]() {
      //op.setSymName(implName);
      op->removeAttr("eaac.riscv_staging_kernel_no_const");
      op->setAttr("eaac.riscv_staging_kernel_no_memref", UnitAttr::get(ctx));
    });

    return success();
  }

};



class SplitLLVMFromEAACPass
    : public impl::SplitLLVMFromEAACBase<SplitLLVMFromEAACPass> {
public:
  using SplitLLVMFromEAACBase::SplitLLVMFromEAACBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();


    RewritePatternSet patterns(ctx);
    patterns.add<CopyConstants>(ctx);
    patterns.add<RemoveMemrefArgs>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createSplitLLVMFromEAACPass() {
  return std::make_unique<SplitLLVMFromEAACPass>();
}

} // namespace eaac
} // namespace mlir
