//===- Dialect.cpp - EAAC dialect implementation --------------------------===//
//
// Implements the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::eaac;

//===----------------------------------------------------------------------===//
// TableGen'd dialect definition
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "eaac/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SemAllocOp
//===----------------------------------------------------------------------===//

void SemAllocOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  setNameFn(getSemaphore(), "sem");
}

//===----------------------------------------------------------------------===//
// DmaStartOp
//===----------------------------------------------------------------------===//

void DmaStartOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       /*stage=*/0, /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       /*stage=*/0, /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}

#define GET_OP_CLASSES
#include "eaac/Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// EAAC Dialect
//===----------------------------------------------------------------------===//

void EAACDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "eaac/Types.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "eaac/Ops.cpp.inc"
      >();
}
