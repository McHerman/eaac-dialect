add_mlir_library(EAACTransforms
  CollectAllocDealloc.cpp
  MemoryAlloc.cpp
  MemRefLivenessAnalysis.cpp
  StreamingChannelAnalysis.cpp
  LocalStaging.cpp
  LowerCopyToDma.cpp
  EncodeDependencies.cpp
  FindAsyncDependency.cpp
  FindAliasDependency.cpp
  InsertRequire.cpp
  LowerAsyncToSemaphore.cpp
  LinalgToEaac.cpp
  InsertLoadStore.cpp
  AssignSemaphoreAddresses.cpp

  DEPENDS
  EAACPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRAsyncDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRAnalysis
)
