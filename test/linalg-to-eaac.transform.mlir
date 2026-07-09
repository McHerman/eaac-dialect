// Transform script that replaces linalg ops with EAAC equivalents.
// Mirrors the --convert-linalg-to-eaac pass but expressed as Transform ops.
// Run with: --transform-interpreter="transform-library-paths=test/linalg-to-eaac.transform.mlir"
// Must run after bufferization so operands are memrefs.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op) {

    // Erase linalg.fill ops that initialize freshly allocated buffers;
    // the downstream compute op overwrites the whole buffer anyway.
    %fills = transform.structured.match ops{["RiscvExecuteOp"]} in %root
        : (!transform.any_op) -> !transform.any_op
    transform.eaac.erase_redundant_fill %fills : !transform.any_op

    // Replace each linalg.matmul with an eaac.matmul, validating the
    // 128x128x128 tile-size limit in the process.
    %mms = transform.structured.match ops{["linalg.matmul"]} in %root
        : (!transform.any_op) -> !transform.any_op
    %eaac_mms = transform.eaac.convert_matmul %mms
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
