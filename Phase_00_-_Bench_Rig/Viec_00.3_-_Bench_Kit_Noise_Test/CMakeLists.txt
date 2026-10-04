cmake_minimum_required(VERSION 3.28)
project(bench-kit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# x86-64-v3 = AVX2 + FMA: P-core và E-core chạy cùng một mã máy, và flags ghi được vào env report
# (khác -march=native, vốn không nói lên máy nào đã build).
set(BENCH_ARCH_FLAGS "-march=x86-64-v3" CACHE STRING "Cờ kiến trúc cho benchmark")

add_library(bench INTERFACE)
target_include_directories(bench INTERFACE include)

add_executable(saxpy noise/saxpy.cpp)
target_link_libraries(saxpy PRIVATE bench)
separate_arguments(_arch_flags NATIVE_COMMAND "${BENCH_ARCH_FLAGS}")
target_compile_options(saxpy PRIVATE -Wall -Wextra ${_arch_flags})
set_target_properties(saxpy PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/noise")

string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
set(_env
  "BENCH_CXX=${CMAKE_CXX_COMPILER}"
  "BENCH_CXXFLAGS=${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}} ${BENCH_ARCH_FLAGS}")

# USES_TERMINAL: in output trực tiếp và không chạy song song với target khác
add_custom_target(noise
  COMMAND "${CMAKE_SOURCE_DIR}/scripts/noise.sh" "$<TARGET_FILE:saxpy>"
  DEPENDS saxpy USES_TERMINAL VERBATIM)
add_custom_target(env-report
  COMMAND "${CMAKE_COMMAND}" -E env ${_env} "${CMAKE_SOURCE_DIR}/scripts/env_report.sh"
  USES_TERMINAL VERBATIM)
add_custom_target(cores-check
  COMMAND "${CMAKE_SOURCE_DIR}/scripts/cores_check.sh"
  USES_TERMINAL VERBATIM)
