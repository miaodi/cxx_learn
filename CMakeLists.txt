cmake_minimum_required(VERSION 3.18)

project(CXX_LEARNING)

set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_definitions(-DNDEBUG)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mfma -mavx2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -S -fverbose-asm")
# include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)

if(COMPILER_SUPPORTS_AVX512F)
  include(CheckCXXSourceRuns)
  set(CMAKE_REQUIRED_FLAGS "-mavx512f")
  check_cxx_source_runs("
    #include <immintrin.h>
    int main() {
      __m512 a = _mm512_set1_ps(1.0f);
      (void)a;
      return 0;
    }
  " AVX512_CPU_SUPPORTS)
  set(CMAKE_REQUIRED_FLAGS "")

  if(AVX512_CPU_SUPPORTS)
    message(STATUS "AVX512 is supported by both compiler and CPU")
    add_definitions(-DAVX512_SUPPORTED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f")
  else()
    message(STATUS "AVX512 is supported by compiler but not by CPU")
  endif()
else()
  message(STATUS "AVX512 is not supported by compiler")
endif()

include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE INTERNAL "Turn off tests")
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG origin/main
)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need main for benchmark::benchmark

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)
        
option(USE_CUDA "Use cuda" OFF)

if(USE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  # find_package(cudss REQUIRED)
endif(USE_CUDA)

configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/config.h.in"
  "${CMAKE_BINARY_DIR}/config.h")
set(CMAKE_INCLUDE_CURRENT_DIR ON)


find_package(OpenMP REQUIRED)
        
MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})

FOREACH(subdir ${SUBDIRS})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/CMakeLists.txt)
                ADD_SUBDIRECTORY(${subdir})
        endif()
ENDFOREACH()