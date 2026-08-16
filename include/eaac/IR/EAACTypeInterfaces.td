//===- EAACTypeInterfaces.td - EAAC type interfaces --------*- tablegen -*-===//
//
// Defines type interfaces for the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_TYPE_INTERFACES
#define EAAC_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

def FunitTypeInterface : TypeInterface<"FunitTypeInterface"> {
  let description = [{
    Interface for hardware functional unit types (e.g. !eaac.gemm,
    !eaac.load, ...), so generic scheduling code can query properties
    common to every unit kind without switching on the concrete type.
  }];
  let cppNamespace = "::mlir::eaac";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the number of pipeline stages of this hardware unit",
      /*retTy=*/"unsigned",
      /*methodName=*/"getPipelineDepth",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/ [{
        return $_type.getPipelineDepth();
      }]
    >,
  ];
}

#endif // EAAC_TYPE_INTERFACES
