add_mlir_library(EAACTransforms
  CollectAllocDealloc.cpp
  MemoryAlloc.cpp

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
