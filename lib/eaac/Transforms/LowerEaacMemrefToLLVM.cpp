#include "eaac/Passes.h"

#include "eaac/Dialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "eaac-lower-memref-to-llvm"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_LOWEREAACMEMREFTOLLVM
#include "eaac/Passes.h.inc"

namespace {

// TODO
// Find all risv-v kernels.
// Find function dependcies.
// lower memrefs to LLVM struct
// fix allocated pointer to eaac address
// after lower-func-to-llvm, figure out how to stitch memrefs together
//
// Steal the following pattern for the the lowering:

/*
/// Convert a memref type into a list of LLVM IR types that will form the
/// memref descriptor. The result contains the following types:
///  1. The pointer to the allocated data buffer, followed by
///  2. The pointer to the aligned data buffer, followed by
///  3. A lowered `index`-type integer containing the distance between the
///  beginning of the buffer and the first element to be accessed through the
///  view, followed by
///  4. An array containing as many `index`-type integers as the rank of the
///  MemRef: the array represents the size, in number of elements, of the memref
///  along the given dimension. For constant MemRef dimensions, the
///  corresponding size entry is a constant whose runtime value must match the
///  static value, followed by
///  5. A second array containing as many `index`-type integers as the rank of
///  the MemRef: the second array represents the "stride" (in tensor abstraction
///  sense), i.e. the number of consecutive elements of the underlying buffer.
///  TODO: add assertions for the static cases.
///
///  If `unpackAggregates` is set to true, the arrays described in (4) and (5)
///  are expanded into individual index-type elements.
///
///  template <typename Elem, typename Index, size_t Rank>
///  struct {
///    Elem *allocatedPtr;
///    Elem *alignedPtr;
///    Index offset;
///    Index sizes[Rank]; // omitted when rank == 0
///    Index strides[Rank]; // omitted when rank == 0
///  };
SmallVector<Type, 5>
LLVMTypeConverter::getMemRefDescriptorFields(MemRefType type,
                                             bool unpackAggregates) const {
  if (!type.isStrided()) {
    emitError(
        UnknownLoc::get(type.getContext()),
        "conversion to strided form failed either due to non-strided layout "
        "maps (which should have been normalized away) or other reasons");
    return {};
  }

  Type elementType = convertType(type.getElementType());
  if (!elementType)
    return {};

  FailureOr<unsigned> addressSpace = getMemRefAddressSpace(type);
  if (failed(addressSpace)) {
    emitError(UnknownLoc::get(type.getContext()),
              "conversion of memref memory space ")
        << type.getMemorySpace()
        << " to integer address space "
           "failed. Consider adding memory space conversions.";
    return {};
  }
  auto ptrTy = LLVM::LLVMPointerType::get(type.getContext(), *addressSpace);

  auto indexTy = getIndexType();

  SmallVector<Type, 5> results = {ptrTy, ptrTy, indexTy};
  auto rank = type.getRank();
  if (rank == 0)
    return results;

  if (unpackAggregates)
    results.insert(results.end(), 2 * rank, indexTy);
  else
    results.insert(results.end(), 2, LLVM::LLVMArrayType::get(indexTy, rank));
  return results;
}
*/

// Fires inside RISC-V staging wrapper functions only.
//
// RiscvKernelToLLVM leaves an unrealized_conversion_cast from each
// memref<..., #eaac.mem<tier, offset>> argument to the corresponding
// !llvm.struct because the staging func.func stays alive (it is legal in the
// target). This pattern replaces that cast with a manually-built memref
// descriptor whose alignedPtr field is inttoptr(offset), so the RISC-V core
// can reach the data at the hardware-assigned address.
struct EaacMemSpaceCastToLLVM : OpRewritePattern<UnrealizedConversionCastOp> {
  EaacMemSpaceCastToLLVM(MLIRContext *ctx, LLVMTypeConverter &tc)
      : OpRewritePattern<UnrealizedConversionCastOp>(ctx), tc(tc) {}

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp castOp,
                                PatternRewriter &rewriter) const override {
    if (castOp.getNumOperands() != 1 || castOp.getNumResults() != 1)
      return failure();

    auto memrefType = dyn_cast<MemRefType>(castOp.getOperand(0).getType());
    if (!memrefType)
      return failure();

    auto memSpace =
        dyn_cast_or_null<eaac::MemSpaceAttr>(memrefType.getMemorySpace());
    if (!memSpace)
      return failure();

    if (!isa<LLVM::LLVMStructType>(castOp.getResult(0).getType()))
      return failure();

    Location loc = castOp.getLoc();
    int64_t hwOffset = memSpace.getOffset() + 0x80000000;

    // Build a pointer from the hardware-assigned base address. Use the same
    // width as the type converter's index type so this matches the target's
    // pointer width (32 bits for RISC-V) rather than always being i64.
    Type indexTy = tc.getIndexType();
    Type ptrTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value offsetVal = rewriter.create<LLVM::ConstantOp>(
        loc, indexTy, rewriter.getIntegerAttr(indexTy, hwOffset));
    Value ptr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrTy, offsetVal);

    // Build the full descriptor using the plain type (identical struct layout,
    // just without the eaac memory space so the type converter can handle it).
    // fromStaticShape sets allocatedPtr = alignedPtr = ptr, offset = 0, and
    // fills in static sizes and row-major strides from the shape.
    auto plainType = MemRefType::get(memrefType.getShape(),
                                     memrefType.getElementType(),
                                     memrefType.getLayout());
    MemRefDescriptor desc =
        MemRefDescriptor::fromStaticShape(rewriter, loc, tc, plainType, ptr);

    rewriter.replaceOp(castOp, Value(desc));
    return success();
  }

  LLVMTypeConverter &tc;
};

class LowerEaacMemrefToLLVMPass
    : public impl::LowerEaacMemrefToLLVMBase<LowerEaacMemrefToLLVMPass> {
public:
  using LowerEaacMemrefToLLVMBase::LowerEaacMemrefToLLVMBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Must match the 32-bit index override used in RiscvKernelToLLVM so the
    // memref descriptor this pattern builds has the same field types (i32
    // offset/sizes/strides) as the one the kernel's own conversion produced.
    LowerToLLVMOptions llvmOptions(ctx);
    llvmOptions.overrideIndexBitwidth(32);
    LLVMTypeConverter typeConverter(ctx, llvmOptions);

    RewritePatternSet patterns(ctx);
    patterns.add<EaacMemSpaceCastToLLVM>(ctx, typeConverter);
    FrozenRewritePatternSet frozen(std::move(patterns));

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (!func->hasAttr("eaac.riscv_staging_kernel"))
        continue;
      if (failed(applyPatternsGreedily(func, frozen)))
        return signalPassFailure();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createLowerEaacMemrefToLLVMPass() {
  return std::make_unique<eaac::LowerEaacMemrefToLLVMPass>();
}

} // namespace eaac
} // namespace mlir
