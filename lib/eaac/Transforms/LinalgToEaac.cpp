//===- LinalgToEaac.cpp - Convert linalg ops to EAAC equivalents ---------===//
//
// Dialect conversion pass that converts linalg operations to EAAC operations.
// Marks the linalg dialect as illegal so unconverted ops produce errors.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "convert-linalg-to-eaac"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_LINALGTOEAAC
#include "eaac/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Remove linalg.fill ops that only initialize a freshly allocated output
/// buffer. These are redundant when the downstream compute op will overwrite
/// the entire buffer anyway.
struct RemoveRedundantFill : OpConversionPattern<linalg::FillOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value output = op.getDpsInits()[0];

    // Only remove fills into freshly allocated buffers. Since the alloc
    // creates the buffer, there cannot be a prior write — the fill is
    // always the first initialization.
    Operation *defOp = output.getDefiningOp();
    if (!defOp || !isa<memref::AllocOp>(defOp))
      return rewriter.notifyMatchFailure(
          op, "fill destination is not a freshly allocated buffer");

    rewriter.eraseOp(op);
    return success();
  }
};

struct MatmulToEaacMatmul : OpConversionPattern<linalg::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // linalg.matmul has ins(A, B) outs(C).
    auto inputs = op.getDpsInputOperands();
    auto outputs = op.getDpsInits();

    if (inputs.size() != 2 || outputs.size() != 1)
      return rewriter.notifyMatchFailure(op, "expected 2 inputs and 1 output");

    Value a = inputs[0]->get();
    Value b = inputs[1]->get();
    Value c = outputs[0];

    auto aType = dyn_cast<MemRefType>(a.getType());
    auto bType = dyn_cast<MemRefType>(b.getType());
    auto cType = dyn_cast<MemRefType>(c.getType());

    if (!aType || !bType || !cType)
      return rewriter.notifyMatchFailure(op, "expected memref operands");

    // A: MxK, B: KxN, C: MxN
    if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "expected rank-2 memrefs");

    int64_t M = aType.getShape()[0];
    int64_t K = aType.getShape()[1];
    int64_t N = bType.getShape()[1];

    if (M > 128 || N > 128 || K > 128)
      return rewriter.notifyMatchFailure(
          op, "dimensions exceed eaac.matmul limits (128x128)");

    MatmulOp::create(rewriter, op.getLoc(), a, b, c);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class LinalgToEaacPass : public impl::LinalgToEaacBase<LinalgToEaacPass> {
public:
  using LinalgToEaacBase::LinalgToEaacBase;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    // EAAC ops are legal, linalg ops are illegal.
    target.addLegalDialect<EAACDialect>();
    target.addIllegalDialect<linalg::LinalgDialect>();

    // Everything else stays as-is.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&getContext());
    patterns.add<RemoveRedundantFill>(&getContext());
    patterns.add<MatmulToEaacMatmul>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createLinalgToEaacPass() {
  return std::make_unique<LinalgToEaacPass>();
}

} // namespace eaac
} // namespace mlir
