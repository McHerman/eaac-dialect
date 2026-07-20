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
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>

#define DEBUG_TYPE "eaac-split-llvm-from-eaac"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_SPLITLLVMFROMEAAC
#include "eaac/Passes.h.inc"

namespace {

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


    rewriter.setInsertionPointToStart(&op.getBody().front());

    // Materialize an equivalent LLVM::ConstantOp for each arg and replace the
    // corresponding argument's uses with it. 

    for (auto [index, constant] : constantOps) {
      auto llvmConstant = rewriter.create<LLVM::ConstantOp>(constant.getLoc(),
                                                              constant.getValue());
      rewriter.replaceAllUsesWith(op.getArgument(index), llvmConstant.getResult());
    }


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

struct RewriteFuncToLLVM : OpRewritePattern<func::FuncOp> {

  using OpRewritePattern::OpRewritePattern;


  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {

    if (!op->hasAttr("eaac.riscv_staging_kernel_no_memref"))
      return failure();

    MLIRContext *ctx = op->getContext();
    

    // Create LLVM function and move body, no arguments
    auto fnType = LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), {});

    rewriter.setInsertionPoint(op);
    auto llvmFunc = rewriter.create<LLVM::LLVMFuncOp>(op->getLoc(),
                                                        op.getSymName(),
                                                        fnType);
    
    llvmFunc->setAttr("eaac.riscv_staging_kernel_no_memref", UnitAttr::get(ctx));
    rewriter.inlineRegionBefore(op.getBody(), llvmFunc.getBody(),
                                llvmFunc.getBody().end());

    Block &entry = llvmFunc.getBody().front();
    auto funcReturn = cast<func::ReturnOp>(entry.getTerminator());
    rewriter.setInsertionPoint(funcReturn);
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(funcReturn, ValueRange{});

    // Every call site becomes an llvm.call to the same symbol name.
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto uses = SymbolTable::getSymbolUses(op, moduleOp);

    if (uses) {
      for (SymbolTable::SymbolUse use : *uses) {
        auto call = dyn_cast<func::CallOp>(use.getUser());
        if (!call)
          continue;

        rewriter.setInsertionPoint(call);
        rewriter.replaceOpWithNewOp<LLVM::CallOp>(
            call, TypeRange{}, llvmFunc.getSymName(), ValueRange{});
      }
    }

    rewriter.eraseOp(op);
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
    patterns.add<RewriteFuncToLLVM>(ctx);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();

    // Every RISC-V kernel is now a plain LLVM::LLVMFuncOp sitting alongside
    // the remaining EAAC IR. Record the program order the eaac IR calls them
    // in before moving their definitions into a throwaway module, so a
    // synthetic `main` there can drive them in that same order.
    SmallVector<LLVM::LLVMFuncOp> kernels(module.getOps<LLVM::LLVMFuncOp>());
    if (kernels.empty())
      return;


    // Collect wrapper functions
    llvm::DenseSet<StringRef> entryPointNames;
    for (LLVM::LLVMFuncOp fn : kernels)
      if (fn->hasAttr("eaac.riscv_staging_kernel_no_memref"))
        entryPointNames.insert(fn.getSymName());

    // Gather callops in order
    SmallVector<StringRef> callOrder;
    module.walk([&](LLVM::CallOp callOp) {
      if (auto callee = callOp.getCallee())
        if (entryPointNames.contains(*callee))
          callOrder.push_back(*callee);
    });

    // Create llvm module
    OpBuilder builder(ctx);
    auto llvmModule = ModuleOp::create(builder, module.getLoc());
    for (LLVM::LLVMFuncOp fn : kernels)
      fn->moveBefore(llvmModule.getBody(), llvmModule.getBody()->end());

    // Synthesize the driver `main` that calls each kernel in that order.
    builder.setInsertionPointToEnd(llvmModule.getBody());
    auto voidType = LLVM::LLVMVoidType::get(ctx);
    auto mainType = LLVM::LLVMFunctionType::get(voidType, {});
    auto mainFn = builder.create<LLVM::LLVMFuncOp>(module.getLoc(), "main", mainType);
    Block *mainBody = mainFn.addEntryBlock(builder);


    // Create callop for each wrapper function in order of use in eaac body
    builder.setInsertionPointToEnd(mainBody);
    for (StringRef callee : callOrder)
      builder.create<LLVM::CallOp>(module.getLoc(), TypeRange{}, callee, ValueRange{});
    builder.create<LLVM::ReturnOp>(module.getLoc(), ValueRange{});

    llvm::LLVMContext llvmCtx;
    std::unique_ptr<llvm::Module> translated =
        translateModuleToLLVMIR(llvmModule, llvmCtx);
    if (!translated) {
      llvmModule->emitError("failed to translate extracted RISC-V kernels to LLVM IR");
      return signalPassFailure();
    }

    if (llvmOutputFile.empty()) {
      module.emitWarning("eaac-split-llvm-from-eaac: llvm-output-file not set, "
                         "discarding extracted kernels");
    } else {
      std::error_code ec;
      llvm::raw_fd_ostream os(llvmOutputFile, ec, llvm::sys::fs::OF_None);
      if (ec) {
        module.emitError() << "failed to open '" << llvmOutputFile
                            << "': " << ec.message();
        return signalPassFailure();
      }
      translated->print(os, nullptr);
    }

    llvmModule->erase();
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createSplitLLVMFromEAACPass() {
  return std::make_unique<SplitLLVMFromEAACPass>();
}

} // namespace eaac
} // namespace mlir
