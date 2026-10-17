cmake_minimum_required(VERSION 3.24)
if(NOT TARGET B-Human)
  message(FATAL_ERROR "Configure B-Human's Make/CMake with CMAKE_PROJECT_INCLUDE set to this adapter's Inject.cmake")
endif()
set(BHUMAN_SOURCE "${BHUMAN_PREFIX}")
set(FLATBUFFERS_SOURCE "" CACHE PATH "FlatBuffers v25.2.10 checkout")
set(FLATC "" CACHE FILEPATH "FlatBuffers v25.2.10 schema compiler")
set(BHUMAN_REVISION "2e3374ba896da8288607f9a1881b876642fe18b8")
execute_process(COMMAND git -C "${BHUMAN_SOURCE}" rev-parse HEAD
  OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
if(NOT revision STREQUAL BHUMAN_REVISION)
  message(FATAL_ERROR "Expected B-Human revision ${BHUMAN_REVISION}, got ${revision}")
endif()
if(NOT EXISTS "${FLATBUFFERS_SOURCE}/include/flatbuffers/flatbuffers.h" OR NOT EXISTS "${FLATC}")
  message(FATAL_ERROR "Provide FLATBUFFERS_SOURCE and FLATC for version 25.2.10")
endif()

# The upstream minimal build omits Qt, physical hardware, audio and deployment tools.
get_property(bhuman_targets DIRECTORY "${BHUMAN_SOURCE}/Make/CMake" PROPERTY BUILDSYSTEM_TARGETS)
foreach(target IN LISTS bhuman_targets)
  # ccache is optional for this adapter, even though upstream enables it on Linux.
  set_property(TARGET ${target} PROPERTY CXX_COMPILER_LAUNCHER "")
endforeach()

# Retain the native team message codec and policy; replace its socket transport.
get_target_property(worker_sources B-Human SOURCES)
list(FILTER worker_sources EXCLUDE REGEX "/TeamMessageChannel\\.cpp$")
set_property(TARGET B-Human PROPERTY SOURCES "${worker_sources}")

set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generated}")
add_custom_command(OUTPUT "${generated}/protocol_generated.h"
  COMMAND "${FLATC}" --cpp --scoped-enums -o "${generated}" "${CMAKE_CURRENT_LIST_DIR}/protocol.fbs"
  DEPENDS "${CMAKE_CURRENT_LIST_DIR}/protocol.fbs" VERBATIM)
add_executable(bhuman-motion-worker "${CMAKE_CURRENT_LIST_DIR}/main.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/MotionInputs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/CognitionInputs.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/TeamTransport.cpp" "${generated}/protocol_generated.h")
# Upstream forces mold for desktop executables; the adapter also supports the system linker.
include(CheckLinkerFlag)
check_linker_flag(CXX "-fuse-ld=mold" BHUMAN_HAS_MOLD)
if(NOT BHUMAN_HAS_MOLD)
  get_target_property(worker_link_options bhuman-motion-worker LINK_OPTIONS)
  list(REMOVE_ITEM worker_link_options "-fuse-ld=mold")
  set_property(TARGET bhuman-motion-worker PROPERTY LINK_OPTIONS "${worker_link_options}")
  message(STATUS "mold is unavailable; using the compiler's default linker for B-Human")
endif()
set_property(TARGET bhuman-motion-worker PROPERTY CXX_COMPILER_LAUNCHER "")
target_compile_features(bhuman-motion-worker PRIVATE cxx_std_20)
target_compile_definitions(bhuman-motion-worker PRIVATE LINUX BHUMAN_REVISION="${BHUMAN_REVISION}")
target_include_directories(bhuman-motion-worker PRIVATE "${generated}" "${FLATBUFFERS_SOURCE}/include")
target_link_libraries(bhuman-motion-worker PRIVATE B-Human Flags::Default)
