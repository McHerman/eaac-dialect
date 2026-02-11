add_mlir_library(EAACTransforms
  CollectAllocDealloc.cpp

  DEPENDS
  EAACPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRAnalysis
)
