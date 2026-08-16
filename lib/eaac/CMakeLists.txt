add_mlir_dialect_library(EAACDialect
  Dialect.cpp
  ScheduleInterfacesImpls.cpp

  DEPENDS
  EAACIncGen
  EAACTypesIncGen
  MLIREAACScheduleInterfacesIncGen
  MLIREAACTypeInterfacesIncGen
  MLIREAACOpInterfacesIncGen
)

target_link_libraries(EAACDialect
  PUBLIC
  MLIRIR
  MLIRSupport
  MLIRDLTIDialect
)

add_subdirectory(Transforms)
add_subdirectory(Target)
add_subdirectory(Transform)
