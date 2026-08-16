//===- Traits.td - EAAC dialect native op traits -----------*- tablegen -*-===//
//
// Defines native op traits for the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_TRAITS
#define EAAC_TRAITS

include "mlir/IR/OpBase.td"

def HardwareUnitTrait : NativeOpTrait<"HardwareUnitTrait"> {
  let cppNamespace = "::mlir::eaac::OpTrait";
}

#endif // EAAC_TRAITS
