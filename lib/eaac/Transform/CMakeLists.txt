add_mlir_library(EAACTransformOps
  EAACTransformOps.cpp
  DialectExtension.cpp

  DEPENDS
  EAACTransformOpsIncGen
  EAACIncGen

  LINK_LIBS PUBLIC
  EAACDialect
  MLIRIR
  MLIRSupport
  MLIRTransformDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
)
