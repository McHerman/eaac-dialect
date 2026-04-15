set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=eaac)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=eaac)
add_public_tablegen_target(EAACIncGen)

# Generate pass declarations and definitions
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EAAC)
add_public_tablegen_target(EAACPassIncGen)
