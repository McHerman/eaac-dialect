//===- EAACOpInterfaces.td - EAAC op interfaces -------------*- tablegen -*-===//
//
// Defines op interfaces for the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_OP_INTERFACES
#define EAAC_OP_INTERFACES

include "mlir/IR/OpBase.td"

def HardwareUnitOpInterface : OpInterface<"HardwareUnitOpInterface"> {
  let description = [{
    Interface for hardware allocation ops. Currently a little superfluos, but added for futureproofing.
  }];
  let cppNamespace = "::mlir::eaac";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the hardware unit type this op allocates",
      /*retTy=*/"::mlir::Type",
      /*methodName=*/"getFunitType",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return $_op.getFunitAttr().getValue();
      }]
    >,
  ];
}

#endif // EAAC_OP_INTERFACES
