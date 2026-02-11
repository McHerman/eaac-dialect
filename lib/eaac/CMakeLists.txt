add_mlir_dialect_library(EAACDialect
  Dialect.cpp

  DEPENDS
  EAACIncGen
)

target_link_libraries(EAACDialect
  PUBLIC
  MLIRIR
  MLIRSupport
)

add_subdirectory(Transforms)
