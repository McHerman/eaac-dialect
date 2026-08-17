//===- Dialect.h - EAAC dialect definition --------------------------------===//
//
// Defines the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_DIALECT_H
#define EAAC_DIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Dialect/Async/IR/Async.h"

#include "eaac/Traits.h"

#include "eaac/Dialect.h.inc"

#include "eaac/EAACEnums.h.inc"

#include "eaac/IR/EAACTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "eaac/Types.h.inc"

#define GET_ATTRDEF_CLASSES
#include "eaac/Attrs.h.inc"

#include "eaac/IR/EAACOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "eaac/Ops.h.inc"

#include "eaac/IR/EAACScheduleInterfaces.h.inc"

namespace mlir::eaac {
/// Registers external models attaching ScheduleInterface to ops.
void registerScheduleOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace mlir::eaac

#endif // EAAC_DIALECT_H
