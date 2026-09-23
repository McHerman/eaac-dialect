//===- Traits.h - EAAC dialect native op traits ---------------*- C++ -*-===//
//
// Defines native C++ op traits for the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_TRAITS_H
#define EAAC_TRAITS_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace eaac {
namespace OpTrait {

/// Marks an op as declaring a hardware unit (e.g. eaac.hw_alloc), so
/// generic scheduling code can find hardware unit allocations regardless of
/// their concrete op type, e.g. `op->hasTrait<HardwareUnitTrait>()`.
template <typename ConcreteType>
class HardwareUnitTrait
    : public ::mlir::OpTrait::TraitBase<ConcreteType, HardwareUnitTrait> {};

} // namespace OpTrait
} // namespace eaac
} // namespace mlir

#endif // EAAC_TRAITS_H
