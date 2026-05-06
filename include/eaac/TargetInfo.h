//===- TargetInfo.h - EAAC target hardware description --------*- C++ -*-===//
//
// Reads target-hardware parameters from well-known module attributes:
//   eaac.tier_capacities     : array<i64: ...>  (level 0 = fastest/smallest)
//   eaac.num_semaphore_pairs : i64
// Falls back to defaults if absent.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_TARGETINFO_H
#define EAAC_TARGETINFO_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir {
namespace eaac {

struct EaacTarget {
  llvm::SmallVector<int64_t, 4> tierCapacities;
  int64_t numSemaphorePairs;
};

constexpr llvm::StringLiteral kTierCapacitiesAttrName = "eaac.tier_capacities";
constexpr llvm::StringLiteral kNumSemaphorePairsAttrName =
    "eaac.num_semaphore_pairs";

/// Read target parameters from the module's discardable attributes. Missing or
/// malformed entries fall back to defaults (and emit a warning when malformed).
EaacTarget getEaacTarget(ModuleOp module);

} // namespace eaac
} // namespace mlir

#endif // EAAC_TARGETINFO_H
