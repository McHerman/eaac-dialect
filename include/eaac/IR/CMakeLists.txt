#add_mlir_dialect(Ops eaac)

add_mlir_interface(EAACScheduleInterfaces)
add_mlir_interface(EAACOpInterfaces)

set(LLVM_TARGET_DEFINITIONS EAACTypeInterfaces.td)
mlir_tablegen(EAACTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(EAACTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_mlir_generic_tablegen_target(MLIREAACTypeInterfacesIncGen)
