cmake_minimum_required(VERSION 3.18)

project(CXX_LEARNING LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra)
  # Only add -Wpedantic for C++ files, not CUDA
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options(/W4)
endif()

# Optimization flags for Release builds
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -funroll-loops -march=native")

# Optional: verbose assembly output (uncomment if needed)
# add_compile_options(-S -fverbose-asm)

# Optional: verbose assembly output (uncomment if needed)
# add_compile_options(-S -fverbose-asm)

# ============================================================================
# SIMD Support Detection
# ============================================================================

include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

# Function to check SIMD support
function(check_simd_support FLAG INSTRUCTION_SET COMPILE_VAR RUNTIME_VAR TEST_CODE)
  check_cxx_compiler_flag(${FLAG} ${COMPILE_VAR})
  
  if(${COMPILE_VAR})
    set(CMAKE_REQUIRED_FLAGS ${FLAG})
    check_cxx_source_runs("${TEST_CODE}" ${RUNTIME_VAR})
    set(CMAKE_REQUIRED_FLAGS "")
    
    if(${RUNTIME_VAR})
      message(STATUS "${INSTRUCTION_SET} is supported by both compiler and CPU")
      set(${RUNTIME_VAR} TRUE PARENT_SCOPE)
    else()
      message(STATUS "${INSTRUCTION_SET} is supported by compiler but not by CPU")
      set(${RUNTIME_VAR} FALSE PARENT_SCOPE)
    endif()
  else()
    message(STATUS "${INSTRUCTION_SET} is not supported by compiler")
    set(${RUNTIME_VAR} FALSE PARENT_SCOPE)
  endif()
endfunction()

# Check FMA support
check_simd_support(
  "-mfma" "FMA" COMPILER_SUPPORTS_FMA FMA_CPU_SUPPORTS
  "
  #include <immintrin.h>
  int main() {
    __m128d a = _mm_set1_pd(1.0);
    __m128d b = _mm_set1_pd(2.0);
    __m128d c = _mm_fmadd_pd(a, b, a);
    (void)c;
    return 0;
  }
  "
)

# Check AVX2 support
check_simd_support(
  "-mavx2" "AVX2" COMPILER_SUPPORTS_AVX2 AVX2_CPU_SUPPORTS
  "
  #include <immintrin.h>
  int main() {
    __m256 a = _mm256_set1_ps(1.0f);
    (void)a;
    return 0;
  }
  "
)

# Check AVX512 support
check_simd_support(
  "-mavx512f" "AVX512" COMPILER_SUPPORTS_AVX512F AVX512_CPU_SUPPORTS
  "
  #include <immintrin.h>
  int main() {
    __m512 a = _mm512_set1_ps(1.0f);
    (void)a;
    return 0;
  }
  "
)

# ============================================================================
# Helper function to add SIMD flags to targets
# ============================================================================

function(target_enable_simd TARGET_NAME)
  if(FMA_CPU_SUPPORTS)
    target_compile_options(${TARGET_NAME} PRIVATE -mfma)
    target_compile_definitions(${TARGET_NAME} PRIVATE FMA_SUPPORTED)
  endif()
  
  if(AVX2_CPU_SUPPORTS)
    target_compile_options(${TARGET_NAME} PRIVATE -mavx2)
    target_compile_definitions(${TARGET_NAME} PRIVATE AVX2_SUPPORTED)
  endif()
  
  if(AVX512_CPU_SUPPORTS)
    target_compile_options(${TARGET_NAME} PRIVATE -mavx512f)
    target_compile_definitions(${TARGET_NAME} PRIVATE AVX512_SUPPORTED)
  endif()
endfunction()

# ============================================================================
# External Dependencies
# ============================================================================

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

include(FetchContent)

# Google Test
set(INSTALL_GTEST OFF CACHE BOOL "Disable GTest installation" FORCE)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0  # Use stable release instead of main
  GIT_SHALLOW TRUE
)

# Google Benchmark
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Turn off benchmark tests" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark installation" FORCE)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3  # Use stable release
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(googletest googlebenchmark)

# ============================================================================
# CUDA Support
# ============================================================================
        
option(USE_CUDA "Enable CUDA support" OFF)

if(USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# ============================================================================
# OpenMP Support
# ============================================================================

find_package(OpenMP REQUIRED)

# ============================================================================
# Configuration File
# ============================================================================

configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/config.h.in"
  "${CMAKE_BINARY_DIR}/config.h"
)

include_directories(${CMAKE_BINARY_DIR})

# ============================================================================
# Add Subdirectories
# ============================================================================
        
macro(SUBDIRLIST result curdir)
  # CONFIGURE_DEPENDS makes CMake check for new directories at build time
  file(GLOB children CONFIGURE_DEPENDS RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})

foreach(subdir ${SUBDIRS})
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
    add_subdirectory(${subdir})
  endif()
endforeach()