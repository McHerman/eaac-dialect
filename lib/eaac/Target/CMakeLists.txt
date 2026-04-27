find_package(Flatbuffers QUIET)

# If cmake package not found, find headers/libs manually
if(NOT Flatbuffers_FOUND AND NOT TARGET flatbuffers::flatbuffers)
  find_path(FLATBUFFERS_INCLUDE_DIR flatbuffers/flatbuffers.h)
  find_library(FLATBUFFERS_LIBRARY NAMES flatbuffers)
endif()

add_mlir_library(EAACTarget
  EAACToFlatbuffer.cpp

  DEPENDS
  EAACFlatBufferGen
  EAACIncGen
)

target_include_directories(EAACTarget
  PRIVATE
  ${PROJECT_BINARY_DIR}/include/eaac/Target  # for eaac_program_generated.h
)

if(FLATBUFFERS_INCLUDE_DIR)
  target_include_directories(EAACTarget PUBLIC ${FLATBUFFERS_INCLUDE_DIR})
endif()

target_link_libraries(EAACTarget
  PUBLIC
  EAACDialect
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRLinalgDialect
)

if(FLATBUFFERS_LIBRARY)
  target_link_libraries(EAACTarget PUBLIC ${FLATBUFFERS_LIBRARY})
elseif(TARGET flatbuffers::flatbuffers)
  target_link_libraries(EAACTarget PUBLIC flatbuffers::flatbuffers)
endif()
