# Find flatc compiler
find_program(FLATC flatc REQUIRED)

# Generate C++ header from FlatBuffer schema
set(FBS_SCHEMA ${CMAKE_CURRENT_SOURCE_DIR}/eaac_program.fbs)
set(FBS_GENERATED_HEADER ${CMAKE_CURRENT_BINARY_DIR}/eaac_program_generated.h)

add_custom_command(
  OUTPUT ${FBS_GENERATED_HEADER}
  COMMAND ${FLATC} --cpp -o ${CMAKE_CURRENT_BINARY_DIR} ${FBS_SCHEMA}
  DEPENDS ${FBS_SCHEMA}
  COMMENT "Generating FlatBuffer header from eaac_program.fbs"
)

add_custom_target(EAACFlatBufferGen DEPENDS ${FBS_GENERATED_HEADER})
