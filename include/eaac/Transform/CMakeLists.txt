set(LLVM_TARGET_DEFINITIONS EAACTransformOps.td)
mlir_tablegen(EAACTransformOps.h.inc -gen-op-decls)
mlir_tablegen(EAACTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(EAACTransformOpsIncGen)
