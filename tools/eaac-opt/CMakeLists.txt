set(LLVM_LINK_COMPONENTS
  Core
  Support
)

add_llvm_executable(eaac-opt
  eaac-opt.cpp

  DEPENDS
  EAACIncGen
  EAACPassIncGen
)

target_link_libraries(eaac-opt
  PRIVATE
  EAACDialect
  EAACTransforms
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRMemRefDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
)
