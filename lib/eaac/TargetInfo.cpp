//===- TargetInfo.cpp - EAAC target hardware description ----------------===//

#include "eaac/TargetInfo.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace eaac {

// Default hardware parameters. Override per-module via the well-known module
// attributes declared in TargetInfo.h.
static constexpr int64_t kDefaultTierCapacities[] = {49152, 147456, 16777216};
static constexpr int64_t kDefaultNumSemaphorePairs = 16;

EaacTarget getEaacTarget(ModuleOp module) {
  EaacTarget target;

  // Tier capacities.
  Attribute capAttr = module->getAttr(kTierCapacitiesAttrName);
  if (auto arr = dyn_cast_or_null<DenseI64ArrayAttr>(capAttr)) {
    target.tierCapacities.assign(arr.asArrayRef().begin(),
                                 arr.asArrayRef().end());
  } else {
    if (capAttr)
      module.emitWarning() << "ignoring '" << kTierCapacitiesAttrName
                           << "': expected array<i64>, got " << capAttr;
    target.tierCapacities.assign(std::begin(kDefaultTierCapacities),
                                 std::end(kDefaultTierCapacities));
  }

  // Semaphore pair count.
  Attribute semAttr = module->getAttr(kNumSemaphorePairsAttrName);
  if (auto i = dyn_cast_or_null<IntegerAttr>(semAttr)) {
    target.numSemaphorePairs = i.getInt();
  } else {
    if (semAttr)
      module.emitWarning() << "ignoring '" << kNumSemaphorePairsAttrName
                           << "': expected integer, got " << semAttr;
    target.numSemaphorePairs = kDefaultNumSemaphorePairs;
  }

  return target;
}

} // namespace eaac
} // namespace mlir
