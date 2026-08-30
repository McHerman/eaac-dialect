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
// TableGen'd type interface definitions
//===----------------------------------------------------------------------===//

#include "eaac/IR/EAACTypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "eaac/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "eaac/Attrs.cpp.inc"

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

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

void MatmulOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc0Mutable(),
                       /*stage=*/0, /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), &getSrc1Mutable(),
                       /*stage=*/0, /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       /*stage=*/0, /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

void LoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       /*stage=*/0, /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

void StoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       /*stage=*/0, /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
  // Mark as writing to an external resource so the op is not DCE'd.
  effects.emplace_back(MemoryEffects::Write::get(),
                       /*stage=*/0, /*effectOnFullRegion=*/true,
                       SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ExecuteOp
//===----------------------------------------------------------------------===//

Operation *ExecuteOp::getPayloadOp() {
  Operation *payload = nullptr;
  for (Operation &op : getBody().getOps())
    if (!isa<SemRequireOp, SemAcquireOp>(op))
      payload = &op;
  return payload;
}

SemAcquireOp ExecuteOp::getAcquireOp() {
  auto ops = getBody().getOps<SemAcquireOp>();
  return ops.empty() ? nullptr : *ops.begin();
}

SmallVector<SemRequireOp, 4> ExecuteOp::getRequireOps() {
  return llvm::to_vector(getBody().getOps<SemRequireOp>());
}

LogicalResult ExecuteOp::verify() {
  int64_t payloadCount = 0;
  int64_t acquireCount = 0;
  for (Operation &op : getBody().getOps()) {
    if (isa<SemAcquireOp>(op))
      acquireCount++;
    else if (!isa<SemRequireOp>(op))
      payloadCount++;
  }
  if (payloadCount != 1)
    return emitOpError("expects exactly one non-semaphore op in body, found ")
           << payloadCount;
  if (acquireCount > 1)
    return emitOpError("expects at most one sem_acquire in body, found ")
           << acquireCount;
  return success();
}

//===----------------------------------------------------------------------===//
// SemRequireOp
//===----------------------------------------------------------------------===//

SemAcquireOp SemRequireOp::getProducer() {
  for (Operation *user : getSemaphore().getUsers())
    if (auto acquire = dyn_cast<SemAcquireOp>(user))
      return acquire;
  return nullptr;
}

#include "eaac/EAACEnums.cpp.inc"

#include "eaac/IR/EAACOpInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "eaac/Ops.cpp.inc"

#include "eaac/IR/EAACScheduleInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// EAAC Dialect
//===----------------------------------------------------------------------===//

void EAACDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "eaac/Types.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "eaac/Attrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "eaac/Ops.cpp.inc"
      >();
}
