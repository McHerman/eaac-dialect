add_mlir_library(EAACTransforms
  CollectAllocDealloc.cpp
  MemoryAlloc.cpp
  MemRefLivenessAnalysis.cpp
  LocalStaging.cpp
  LowerCopyToDma.cpp

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
