include "mlir/IR/OpBase.td"

def ScheduleInterface : OpInterface<"ScheduleInterface"> {
  let description = [{
    Interface for querying scheduling information for given operation
  }];
  let cppNamespace = "::mlir::eaac";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Check if compatible hardware unit is available",
      /*retTy=*/"::mlir::Operation *",
      /*methodName=*/"checkSchedule",
      /*args=*/(ins "Region *":$region),
      /*methodBody*/[{}],
      /*defaultImplementation=*/ [{
        return nullptr;
      }]
    >,
  ];
}
