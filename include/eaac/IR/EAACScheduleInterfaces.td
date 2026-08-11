include "mlir/IR/OpBase.td"

def AffineReadOpInterface : OpInterface<"AffineReadOpInterface"> {
  let description = [{
    Interface for querying scheduling information for given operation
  }];
  let cppNamespace = "::mlir::affine";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Check if compatible hardware unit is available",
      /*retTy=*/"::mlir::Value",
      /*methodName=*/"checkSchedule",
      /*args=*/(ins "Region *":$region),
      /*methodBody*/[{}],
      /*defaultImplementation=*/ [{
        return null;
      }]
    >,
  ];
}
