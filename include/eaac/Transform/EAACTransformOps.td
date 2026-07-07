//===- EAACTransformOps.td - EAAC transform ops ------------*- tablegen -*-===//
//
// Defines Transform dialect extension ops for lowering kernels into EAAC ops.
// These ops express the kernel-specific lowering step (e.g. linalg → eaac)
// separately from the data-orchestration passes that wrap it.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_TRANSFORM_OPS
#define EAAC_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def EAACConvertMatmulOp
    : Op<Transform_Dialect, "eaac.convert_matmul",
         [TransformEachOpTrait, TransformOpInterface,
          FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface]> {
  let summary = "Convert a linalg.matmul to eaac.matmul";
  let description = [{
    Takes a handle to a `linalg.matmul` operation and replaces it in-place
    with an `eaac.matmul` operation. Validates that all dimensions fit within
    the hardware 128x128x128 tile limit.

    Returns a handle to the new `eaac.matmul`.

    Example:
    ```mlir
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %eaac_mm = transform.eaac.convert_matmul %mm
        : (!transform.any_op) -> !transform.any_op
    ```
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results  = (outs TransformHandleTypeInterface:$result);
  let assemblyFormat = [{
    $target attr-dict `:` functional-type($target, $result)
  }];
  let cppNamespace = "::mlir::eaac::transform";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

def EAACEraseRedundantFillOp
    : Op<Transform_Dialect, "eaac.erase_redundant_fill",
         [TransformEachOpTrait, TransformOpInterface,
          FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface]> {
  let summary = "Erase a linalg.fill that initializes a freshly allocated buffer";
  let description = [{
    Takes a handle to a `linalg.fill` whose output operand is defined by a
    `memref.alloc`. Erases the fill because the downstream compute op will
    overwrite the entire buffer anyway.

    Emits a silenceable failure if the fill destination is not a fresh alloc,
    allowing the caller to ignore it or propagate the error.

    Example:
    ```mlir
    %fills = transform.structured.match ops{["linalg.fill"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.eaac.erase_redundant_fill %fills : !transform.any_op
    ```
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);
  let results  = (outs);
  let assemblyFormat = [{
    $target attr-dict `:` type($target)
  }];
  let cppNamespace = "::mlir::eaac::transform";

  let extraClassDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure applyToOne(
        ::mlir::transform::TransformRewriter &rewriter,
        ::mlir::Operation *target,
        ::mlir::transform::ApplyToEachResultList &results,
        ::mlir::transform::TransformState &state);
  }];
}

#endif // EAAC_TRANSFORM_OPS
