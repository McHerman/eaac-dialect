//===- DialectExtension.cpp - EAAC Transform dialect extension ------------===//
//
// Registers the EAAC transform ops as an extension of the Transform dialect.
//
//===----------------------------------------------------------------------===//

#include "eaac/Transform/EAACTransformOps.h"
#include "eaac/Dialect.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;

namespace {
class EAACTransformDialectExtension
    : public transform::TransformDialectExtension<
          EAACTransformDialectExtension> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EAACTransformDialectExtension)

  using Base::Base;

  void init() {
    declareDependentDialect<eaac::EAACDialect>();
    declareDependentDialect<linalg::LinalgDialect>();
    declareDependentDialect<memref::MemRefDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "eaac/Transform/EAACTransformOps.cpp.inc"
        >();
  }
};
} // namespace

void mlir::eaac::registerEAACTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<EAACTransformDialectExtension>();
}
