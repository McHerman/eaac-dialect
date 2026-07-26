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
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/BitVector.h"
#include <cstdint>

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


static int64_t readNumSemaphoreGenerations(ModuleOp module) {
  auto sysSpec = dyn_cast_or_null<TargetSystemSpecAttr>(
      module->getAttr(DLTIDialect::kTargetSystemDescAttrName));
  if (!sysSpec)
    return 1;
  auto deviceId = StringAttr::get(module.getContext(), "EAAC");
  std::optional<TargetDeviceSpecInterface> deviceSpec =
      sysSpec.getDeviceSpecForDeviceID(deviceId);
  if (!deviceSpec)
    return 1;
  for (DataLayoutEntryInterface entry : (*deviceSpec).getEntries()) {
    auto entryKey = dyn_cast<StringAttr>(entry.getKey());
    if (!entryKey || entryKey.getValue() != "num_semaphore_generations")
      continue;
    auto i = dyn_cast<IntegerAttr>(entry.getValue());
    if (!i)
      return 1;
    return i.getInt();
  }
  return 1;
}


static unsigned computeGenWidth(int64_t numGenerations) {
  if (numGenerations <= 1)
    return 0;
  unsigned w = 0;
  uint64_t n = static_cast<uint64_t>(numGenerations - 1);
  while (n) {
    n >>= 1;
    ++w;
  }
  return w;
}


struct ConvertEAACAcquire : OpConversionPattern<eaac::SemAcquireOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(eaac::SemAcquireOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // The following lowers the acquire operation into a llvm kernel that first reads the state of the semaphroe, acquires and release a block of size stepSize

    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc || !parentFunc->hasAttr("eaac.riscv_kernel_impl"))
      return failure();

    Value semVal = op.getOperand(0);
    auto semType = dyn_cast<SemaphoreType>(semVal.getType());
    if (!semType)
      return failure();

    Location loc = op.getLoc();
    int64_t semAddr = semType.getAddr();
    int64_t semGen = semType.getGen();

    int64_t stepSize = op.getStepSize();

    // calculate machine addr
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();

    //int genWidth = computeGenWidth(readNumSemaphoreGenerations(moduleOp));
    int semNumGen = readNumSemaphoreGenerations(moduleOp);
    int genWidth = computeGenWidth(semNumGen);


    LLVM_DEBUG(llvm::dbgs()
               << "sem num of gen " << semNumGen 
               << " sem gen widht " << genWidth
               << " sem gen  " << semGen 
               << "\n");

    // Semaphoreoffset + semaddr * 8 (2 semaphores of 2 bytes, 2 ports = 8)
    int64_t semAddressFull = ((semAddr * 4) << genWidth) + semGen;
    int64_t hwAddressFull = 0x4000 + semAddressFull;

    int64_t semAddressEmpty = ((semAddr * 4 + 1) << genWidth) + semGen;
    int64_t hwAddressEmpty = 0x4000 + semAddressEmpty;

    LLVM_DEBUG(llvm::dbgs()
               << "sem addr full " << semAddressFull 
               << "sem addr empty " << semAddressEmpty 
               << "\n");

    // RISC-V target is fixed 32-bit, so the index width is hardcoded rather
    // than pulled from a type converter.
    Type indexTy = rewriter.getIntegerType(32);
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Create pointers to empty and full semaphores
    Value offsetVal = LLVM::ConstantOp::create(rewriter, loc, indexTy, rewriter.getIntegerAttr(indexTy, hwAddressFull));
    Value ptrFull = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, offsetVal);

    Value offsetValEmpty = LLVM::ConstantOp::create(rewriter, loc, indexTy, rewriter.getIntegerAttr(indexTy, hwAddressEmpty));
    Value ptrEmpty = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, offsetValEmpty);

    // create blocks, split original block, entry and acquireblock sits before main ops, loop block sits before acquire, release sits after main ops
    mlir::Block *entryBlock = op->getBlock();
    mlir::Block *acquireBlock =
        rewriter.splitBlock(entryBlock, Block::iterator(op));
    auto *loopBlock = rewriter.createBlock(acquireBlock);

    rewriter.setInsertionPointToEnd(entryBlock);
    LLVM::BrOp::create(rewriter, loc, loopBlock);

    rewriter.setInsertionPointToStart(loopBlock);

    auto val = LLVM::LoadOp::create(rewriter,loc,indexTy,ptrEmpty);

    Value stepConst = LLVM::ConstantOp::create(
        rewriter, loc, indexTy, rewriter.getIntegerAttr(indexTy, stepSize));

    auto cmp = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::eq, val, stepConst);
    LLVM::CondBrOp::create(
        rewriter, loc, cmp, acquireBlock, ValueRange{}, loopBlock, ValueRange{});

    // Acquire: physically first in acquireBlock, i.e. before the main op.
    rewriter.setInsertionPointToStart(acquireBlock);

    Value stepConstN = LLVM::ConstantOp::create(
        rewriter, loc, indexTy, rewriter.getIntegerAttr(indexTy, -stepSize));

    auto acquire = LLVM::AtomicRMWOp::create(
        rewriter, loc, LLVM::AtomicBinOp::add, ptrEmpty, stepConstN,
        LLVM::AtomicOrdering::acquire);

    // Release: inserted right before acquireBlock's terminator. Since the
    // terminator (e.g. func.return) is always last, this runs after
    // everything else in the block, including the main computation, without
    // needing to know where that computation ends.
    rewriter.setInsertionPoint(acquireBlock->getTerminator());

    auto release = LLVM::AtomicRMWOp::create(
        rewriter, loc, LLVM::AtomicBinOp::add, ptrFull, stepConst,
        LLVM::AtomicOrdering::release);

    // Acquiring is a synchronization handshake; the memref value itself
    // passes through unchanged once the wait loop is satisfied.
    rewriter.replaceOp(op, adaptor.getMemref());

    return success();
  }
};

struct ConvertEAACRequire : OpConversionPattern<eaac::SemRequireOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(eaac::SemRequireOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Does mostly the same as the acquire, but reads from different addresses 
    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (!parentFunc || !parentFunc->hasAttr("eaac.riscv_kernel_impl"))
      return failure();

    Value semVal = op.getOperand(0);
    auto semType = dyn_cast<SemaphoreType>(semVal.getType());
    if (!semType)
      return failure();

    Location loc = op.getLoc();
    int64_t semAddr = semType.getAddr();
    int64_t semGen = semType.getGen();

    int64_t stepSize = op.getStepSize();

    // calculate machine addr
    ModuleOp moduleOp = op->getParentOfType<ModuleOp>();

    int genWidth = computeGenWidth(readNumSemaphoreGenerations(moduleOp));

    // Semaphoreoffset + semaddr * 8 (2 semaphores of 2 bytes, 2 ports = 8) added port offset
    int64_t hwAddressFull = 0x4000 + ((semAddr * 4 + 2) << genWidth) + semGen;
    int64_t hwAddressEmpty = 0x4000 + ((semAddr * 4 + 2 + 1) << genWidth) + semGen;

    // RISC-V target is fixed 32-bit, so the index width is hardcoded rather
    // than pulled from a type converter.
    Type indexTy = rewriter.getIntegerType(32);
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());

    // Create pointers to empty and full semaphores
    Value offsetVal = LLVM::ConstantOp::create(rewriter, loc, indexTy, rewriter.getIntegerAttr(indexTy, hwAddressFull));
    Value ptrFull = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, offsetVal);

    Value offsetValEmpty = LLVM::ConstantOp::create(rewriter, loc, indexTy, rewriter.getIntegerAttr(indexTy, hwAddressEmpty));
    Value ptrEmpty = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, offsetValEmpty);

    // create blocks, split original block, entry and acquireblock sits before main ops, loop block sits before acquire, release sits after main ops
    mlir::Block *entryBlock = op->getBlock();
    mlir::Block *acquireBlock =
        rewriter.splitBlock(entryBlock, Block::iterator(op));
    auto *loopBlock = rewriter.createBlock(acquireBlock);

    rewriter.setInsertionPointToEnd(entryBlock);
    LLVM::BrOp::create(rewriter, loc, loopBlock);

    rewriter.setInsertionPointToStart(loopBlock);

    auto val = LLVM::LoadOp::create(rewriter,loc,indexTy,ptrFull);

    Value stepConst = LLVM::ConstantOp::create(
        rewriter, loc, indexTy, rewriter.getIntegerAttr(indexTy, stepSize));

    auto cmp = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::eq, val, stepConst);
    LLVM::CondBrOp::create(
        rewriter, loc, cmp, acquireBlock, ValueRange{}, loopBlock, ValueRange{});

    // Acquire: physically first in acquireBlock, i.e. before the main op.
    rewriter.setInsertionPointToStart(acquireBlock);

    Value stepConstN = LLVM::ConstantOp::create(
        rewriter, loc, indexTy, rewriter.getIntegerAttr(indexTy, -stepSize));

    // We acquite the empty semaphore
    auto acquire = LLVM::AtomicRMWOp::create(
        rewriter, loc, LLVM::AtomicBinOp::add, ptrFull, stepConstN,
        LLVM::AtomicOrdering::acquire);

    // Release: inserted right before acquireBlock's terminator. Since the
    // terminator (e.g. func.return) is always last, this runs after
    // everything else in the block, including the main computation, without
    // needing to know where that computation ends.


    // We release into the full semaphore
    rewriter.setInsertionPoint(acquireBlock->getTerminator());

    auto release = LLVM::AtomicRMWOp::create(
        rewriter, loc, LLVM::AtomicBinOp::add, ptrEmpty, stepConst,
        LLVM::AtomicOrdering::release);

    // Acquiring is a synchronization handshake; the memref value itself
    // passes through unchanged once the wait loop is satisfied.
    rewriter.replaceOp(op, adaptor.getMemref());

    return success();
  }
};

// Cleans up function signature and call sites.
struct FixFunctionSignature : OpRewritePattern<func::FuncOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr("eaac.riscv_staging_kernel"))
      return failure();

    func::CallOp implCall;
    for (Operation &bodyOp : op.getFunctionBody().front())
      if (auto call = dyn_cast<func::CallOp>(bodyOp)) {
        implCall = call;
        break;
      }
    if (!implCall)
      return failure();

    auto impl = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
        implCall, implCall.getCalleeAttr());
    if (!impl)
      return failure();

    llvm::BitVector deadArgs(impl.getNumArguments());
    for (BlockArgument arg : impl.getArguments())
      if (isa<SemaphoreType>(arg.getType()) && arg.use_empty())
        deadArgs.set(arg.getArgNumber());
    if (deadArgs.none())
      return failure();

    // Drop the matching operand from every call site of the wrapper (e.g.
    // from @main) before the wrapper's signature changes underneath them.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (auto uses = SymbolTable::getSymbolUses(op, module)) {
      for (SymbolTable::SymbolUse use : *uses) {
        auto callOp = dyn_cast<func::CallOp>(use.getUser());
        if (!callOp)
          continue;
        SmallVector<Value> keptOperands;
        for (auto [idx, operand] : llvm::enumerate(callOp.getOperands()))
          if (!deadArgs.test(idx))
            keptOperands.push_back(operand);
        rewriter.modifyOpInPlace(
            callOp, [&] { callOp.getOperandsMutable().assign(keptOperands); });
      }
    }

    // Same for the wrapper's own call into the impl.
    SmallVector<Value> keptCallOperands;
    for (auto [idx, operand] : llvm::enumerate(implCall.getOperands()))
      if (!deadArgs.test(idx))
        keptCallOperands.push_back(operand);
    rewriter.modifyOpInPlace(implCall, [&] {
      implCall.getOperandsMutable().assign(keptCallOperands);
    });

    rewriter.modifyOpInPlace(op, [&] { (void)op.eraseArguments(deadArgs); });
    rewriter.modifyOpInPlace(impl,
                              [&] { (void)impl.eraseArguments(deadArgs); });

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

    // Stage 0.5: Lower eaac.sem_acquire/sem_require to a hardware wait-loop.
    {
      auto isInRiscvKernelImpl = [](Operation *op) {
        auto func = op->getParentOfType<func::FuncOp>();
        return !(func && func->hasAttr("eaac.riscv_kernel_impl"));
      };
      ConversionTarget semTarget(*ctx);
      semTarget.addDynamicallyLegalOp<eaac::SemAcquireOp>(isInRiscvKernelImpl);
      semTarget.addDynamicallyLegalOp<eaac::SemRequireOp>(isInRiscvKernelImpl);
      semTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

      RewritePatternSet semPatterns(ctx);
      semPatterns.add<ConvertEAACAcquire>(ctx);
      semPatterns.add<ConvertEAACRequire>(ctx);
      if (failed(applyPartialConversion(module, semTarget,
                                        std::move(semPatterns))))
        return signalPassFailure();
    }

    // Stage 0.6: The sem_acquire/sem_require ops are gone now, so their
    // semaphore arguments are dead. Remove them from the impl signature, the
    // wrapper's call into the impl, the wrapper signature, and any remaining
    // callers of the wrapper.
    {
      RewritePatternSet fixSigPatterns(ctx);
      fixSigPatterns.add<FixFunctionSignature>(ctx);
      if (failed(applyPatternsGreedily(module, std::move(fixSigPatterns))))
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

    target.addDynamicallyLegalOp<eaac::SemRequireOp>([](eaac::SemRequireOp op) {
      return op->hasAttr("eaac.riscv_kernel_impl");
    });

    target.addDynamicallyLegalOp<eaac::SemAcquireOp>([](eaac::SemAcquireOp op) {
      return op->hasAttr("eaac.riscv_kernel_impl");
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
