set(LLVM_LINK_COMPONENTS
  Core
  Support
)

add_llvm_executable(eaac-translate
  eaac-translate.cpp

  DEPENDS
  EAACIncGen
  EAACFlatBufferGen
)

target_link_libraries(eaac-translate
  PRIVATE
  EAACDialect
  EAACTarget
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTranslateLib
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
)
