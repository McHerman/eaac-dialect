//===- EAACTransformOps.cpp - EAAC transform op implementations -----------===//
//
// Implements the EAAC Transform dialect extension ops.
//
//===----------------------------------------------------------------------===//

#include "eaac/Transform/EAACTransformOps.h"
#include "eaac/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/PatternMatch.h"

#define GET_OP_CLASSES
#include "eaac/Transform/EAACTransformOps.cpp.inc"

using namespace mlir;
using namespace mlir::transform;

namespace mlir {
namespace eaac {
namespace transform {

//===----------------------------------------------------------------------===//
// EAACConvertMatmulOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure EAACConvertMatmulOp::applyToOne(
    TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, TransformState &state) {
  auto matmul = dyn_cast<linalg::MatmulOp>(target);
  if (!matmul)
    return emitSilenceableError()
           << "expected linalg.matmul, got " << target->getName();

  auto inputs = matmul.getDpsInputOperands();
  auto outputs = matmul.getDpsInits();

  if (inputs.size() != 2 || outputs.size() != 1)
    return emitSilenceableError() << "expected 2 inputs and 1 output";

  Value a = inputs[0]->get();
  Value b = inputs[1]->get();
  Value c = outputs[0];

  auto aType = dyn_cast<MemRefType>(a.getType());
  auto bType = dyn_cast<MemRefType>(b.getType());
  auto cType = dyn_cast<MemRefType>(c.getType());

  if (!aType || !bType || !cType)
    return emitSilenceableError() << "expected memref operands";

  if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2)
    return emitSilenceableError() << "expected rank-2 memrefs";

  int64_t M = aType.getShape()[0];
  int64_t K = aType.getShape()[1];
  int64_t N = bType.getShape()[1];

  if (M > 128 || N > 128 || K > 128)
    return emitSilenceableError()
           << "matmul dimensions " << M << "x" << K << "x" << N
           << " exceed eaac.matmul limit (128x128x128)";

  rewriter.setInsertionPoint(matmul);
  auto eaacOp = eaac::MatmulOp::create(rewriter, matmul.getLoc(), a, b, c);
  rewriter.eraseOp(matmul);
  results.push_back(eaacOp);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// EAACEraseRedundantFillOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure EAACEraseRedundantFillOp::applyToOne(
    TransformRewriter &rewriter, Operation *target,
    ApplyToEachResultList &results, TransformState &state) {
  auto fill = dyn_cast<linalg::FillOp>(target);
  if (!fill)
    return emitSilenceableError()
           << "expected linalg.fill, got " << target->getName();

  Value output = fill.getDpsInits()[0];
  Operation *defOp = output.getDefiningOp();
  if (!defOp || !isa<memref::AllocOp>(defOp))
    return emitSilenceableError()
           << "fill destination is not a freshly allocated buffer";

  rewriter.eraseOp(fill);
  return DiagnosedSilenceableFailure::success();
}

} // namespace transform
} // namespace eaac
} // namespace mlir
