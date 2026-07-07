//===- EAACTransformOps.h - EAAC Transform extension ops --------*- C++ -*-===//
//
// Declares the Transform dialect extension ops for EAAC kernel lowering.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_TRANSFORM_EAACTRANSFORMOPS_H
#define EAAC_TRANSFORM_EAACTRANSFORMOPS_H

#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformTypes.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "eaac/Transform/EAACTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace eaac {
void registerEAACTransformDialectExtension(DialectRegistry &registry);
} // namespace eaac
} // namespace mlir

#endif // EAAC_TRANSFORM_EAACTRANSFORMOPS_H
