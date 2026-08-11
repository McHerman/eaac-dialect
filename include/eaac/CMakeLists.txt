set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -dialect=eaac)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -dialect=eaac)
mlir_tablegen(Attrs.h.inc -gen-attrdef-decls -dialect=eaac)
mlir_tablegen(Attrs.cpp.inc -gen-attrdef-defs -dialect=eaac)
add_public_tablegen_target(EAACTypesIncGen)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=eaac)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=eaac)
mlir_tablegen(EAACEnums.h.inc -gen-enum-decls)
mlir_tablegen(EAACEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(EAACIncGen)

# Generate pass declarations and definitions
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name EAAC)
add_public_tablegen_target(EAACPassIncGen)

# FlatBuffer schema generation
add_subdirectory(Target)

add_subdirectory(Transform)

add_subdirectory(IR)
