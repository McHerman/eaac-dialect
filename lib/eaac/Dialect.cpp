//===- Dialect.cpp - EAAC dialect implementation --------------------------===//
//
// Implements the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::eaac;

//===----------------------------------------------------------------------===//
// TableGen'd dialect definition
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "eaac/Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// EAAC Dialect
//===----------------------------------------------------------------------===//

void EAACDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "eaac/Ops.cpp.inc"
      >();
}
