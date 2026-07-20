#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "eaac-riscv-kernel-to-llvm"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_RISCVKERNELTOLLVM
#include "eaac/Passes.h.inc"

namespace {


struct ReplaceMLIRFunctionCall : OpRewritePattern<func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("eaac.riscv_kernel"))
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    FunctionType function_type = op.getFunctionType();
    Location loc = op.getLoc();
    MLIRContext *ctx = op->getContext();

    // Keep the original public name for the staging wrapper so that callers
    // (e.g. @main) need no updates. The implementation is moved under a
    // private __impl_ name and will be fully lowered to llvm.func.
    std::string publicName = op.getSymName().str();
    std::string implName = "__impl_" + publicName;

    // Step 1: rename the kernel in-place to the impl name.
    rewriter.modifyOpInPlace(op, [&]() {
      op.setSymName(implName);
      op->removeAttr("eaac.riscv_kernel");
      op->setAttr("eaac.riscv_kernel_impl", UnitAttr::get(ctx));
    });

    // Step 2: create the staging wrapper that keeps the public name.
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(module.getBody());

    auto staging = func::FuncOp::create(rewriter, loc, publicName, function_type);
    staging->setAttr("eaac.riscv_staging_kernel", UnitAttr::get(ctx));

    Block *body = rewriter.createBlock(
        &staging.getBody(), {}, function_type.getInputs(),
        SmallVector<Location>(function_type.getNumInputs(), loc));

    rewriter.setInsertionPointToStart(body);
    auto call = func::CallOp::create(rewriter, loc, implName,
                                     function_type.getResults(),
                                     body->getArguments());
    func::ReturnOp::create(rewriter, loc, call.getResults());

    return success();
  }
};


class RiscvKernelToLLVMPass
    : public impl::RiscvKernelToLLVMBase<RiscvKernelToLLVMPass> {
public:
  using RiscvKernelToLLVMBase::RiscvKernelToLLVMBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Stage 0: Wrap each riscv kernel in a staging function.
    {
      RewritePatternSet wrapPatterns(ctx);
      wrapPatterns.add<ReplaceMLIRFunctionCall>(ctx);
      if (failed(applyPatternsGreedily(module, std::move(wrapPatterns))))
        return signalPassFailure();
    }

    SmallVector<func::FuncOp> kernels;
    SmallVector<func::FuncOp> stagingFuncs;
    for (func::FuncOp f : module.getOps<func::FuncOp>()) {
      if (f->hasAttr("eaac.riscv_kernel_impl"))
        kernels.push_back(f);
      else if (f->hasAttr("eaac.riscv_staging_kernel"))
        stagingFuncs.push_back(f);
    }

    // Stage 1: linalg → SCF loops. This pass is func.func-anchored so
    // runPipeline can target individual functions without module issues.
    OpPassManager linalgPM("func.func");
    linalgPM.addPass(createConvertLinalgToLoopsPass());
    for (func::FuncOp func : kernels)
      if (failed(runPipeline(linalgPM, func)))
        return signalPassFailure();

    // Stage 2: Lower kernels to llvm.func and update staging wrappers.
    // Both sets are converted in one applyPartialConversion call so there is
    // no intermediate state where a staging wrapper holds a func.call to an
    // already-lowered llvm.func.
    //
    // Kernels → llvm.func (full signature + body conversion).
    // Staging → stays func.func; func.call inside is converted to llvm.call
    //           and memref args are expanded to LLVM components by
    //           CallOpLowering using the type converter.
    // RISC-V target is 32-bit: index-typed values (memref offsets/sizes/
    // strides, SCF loop bounds/induction variables, etc.) must lower to i32,
    // not the LLVM default of i64.
    LowerToLLVMOptions llvmOptions(ctx);
    llvmOptions.overrideIndexBitwidth(32);
    LLVMTypeConverter typeConverter(ctx, llvmOptions);

    // eaac memrefs carry hardware address info in the memory space attribute.
    // For struct-layout purposes they are identical to plain (address-space-0)
    // memrefs; LowerEaacMemrefToLLVM will later replace the generated
    // unrealized_conversion_cast with the correct inttoptr construction.
    //
    // addTypeAttributeConversion covers the structFuncArgTypeConverter path
    // (getMemRefAddressSpace → convertTypeAttribute), which bypasses addConversion.
    // addConversion covers the convertType path used by CallOpLowering.
    typeConverter.addTypeAttributeConversion(
        [](MemRefType, eaac::MemSpaceAttr) -> TypeConverter::AttributeConversionResult {
          return Attribute{}; // null attribute → address space 0
        });
    typeConverter.addConversion([&typeConverter](MemRefType type) -> std::optional<Type> {
      if (!isa_and_nonnull<eaac::MemSpaceAttr>(type.getMemorySpace()))
        return std::nullopt;
      auto plain = MemRefType::get(type.getShape(), type.getElementType(),
                                   type.getLayout());
      return typeConverter.convertType(plain);
    });

    LLVMConversionTarget target(*ctx);
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
      return !op->hasAttr("eaac.riscv_kernel_impl");
    });
    // func.return inside staging wrappers must stay (the func.func stays).
    target.addDynamicallyLegalOp<func::ReturnOp>([](func::ReturnOp op) {
      return op->getParentOp()->hasAttr("eaac.riscv_staging_kernel");
    });

    RewritePatternSet llvmPatterns(ctx);
    populateSCFToControlFlowConversionPatterns(llvmPatterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, llvmPatterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, llvmPatterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, llvmPatterns);
    populateFuncToLLVMConversionPatterns(typeConverter, llvmPatterns);

    FrozenRewritePatternSet frozen(std::move(llvmPatterns));
    SmallVector<Operation *> conversionTargets;
    for (auto f : kernels) conversionTargets.push_back(f.getOperation());
    for (auto f : stagingFuncs) conversionTargets.push_back(f.getOperation());
    if (failed(applyPartialConversion(conversionTargets, target, frozen)))
      return signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createRiscvKernelToLLVMPass() {
  return std::make_unique<RiscvKernelToLLVMPass>();
}

} // namespace eaac
} // namespace mlir
