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

#include "eaac/Dialect.h.inc"

#define GET_OP_CLASSES
#include "eaac/Ops.h.inc"

#endif // EAAC_DIALECT_H
