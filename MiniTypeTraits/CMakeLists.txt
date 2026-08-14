cmake_minimum_required(VERSION 3.18)

set(MINI_TRAITS_MILESTONE 0 CACHE STRING
    "Highest MiniTypeTraits milestone to build (0-5)")
set_property(CACHE MINI_TRAITS_MILESTONE PROPERTY STRINGS 0 1 2 3 4 5)

enable_testing()

if(NOT MINI_TRAITS_MILESTONE MATCHES "^[0-5]$")
  message(FATAL_ERROR "MINI_TRAITS_MILESTONE must be an integer from 0 to 5")
endif()

function(add_mini_traits_test TARGET SOURCE)
  add_executable(${TARGET} ${SOURCE})
  target_compile_features(${TARGET} PRIVATE cxx_std_20)
  target_include_directories(${TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  add_test(NAME ${TARGET} COMMAND ${TARGET})
endfunction()

add_mini_traits_test(mini_traits_scaffold tests/test_scaffold.cpp)

if(MINI_TRAITS_MILESTONE GREATER_EQUAL 1)
  add_mini_traits_test(mini_traits_foundation tests/test_foundation.cpp)
endif()

if(MINI_TRAITS_MILESTONE GREATER_EQUAL 2)
  add_mini_traits_test(mini_traits_transformations tests/test_transformations.cpp)
  add_mini_traits_test(mini_traits_predicates tests/test_predicates.cpp)
  add_mini_traits_test(mini_traits_selection_arithmetic
                       tests/test_selection_arithmetic.cpp)
endif()

if(MINI_TRAITS_MILESTONE GREATER_EQUAL 3)
  add_mini_traits_test(mini_traits_arrays tests/test_arrays.cpp)
  add_mini_traits_test(mini_traits_logical tests/test_logical.cpp)
endif()

if(MINI_TRAITS_MILESTONE GREATER_EQUAL 4)
  add_mini_traits_test(mini_traits_detection tests/test_detection.cpp)
endif()

if(MINI_TRAITS_MILESTONE GREATER_EQUAL 5)
  add_mini_traits_test(mini_traits_realistic_transformations
                       tests/test_realistic_transformations.cpp)
endif()

message(STATUS "MiniTypeTraits milestone ${MINI_TRAITS_MILESTONE} configured")
