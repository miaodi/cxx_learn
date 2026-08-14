cmake_minimum_required(VERSION 3.18)

set(EXPR_AD_MILESTONE 0 CACHE STRING
    "Highest ExpressionTemplatesAD milestone to build (0-6)")
set_property(CACHE EXPR_AD_MILESTONE PROPERTY STRINGS 0 1 2 3 4 5 6)

option(EXPR_AD_USE_REFERENCE
       "Build ExpressionTemplatesAD tests against the reference implementation"
       OFF)

if(NOT "${EXPR_AD_MILESTONE}" MATCHES "^[0-6]$")
  message(FATAL_ERROR "EXPR_AD_MILESTONE must be an integer from 0 to 6")
endif()

if(EXPR_AD_USE_REFERENCE)
  set(EXPR_AD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/solutions/include")
  message(STATUS "ExpressionTemplatesAD is using the reference implementation")
else()
  set(EXPR_AD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
endif()

add_library(expr_ad_course INTERFACE)
target_compile_features(expr_ad_course INTERFACE cxx_std_20)
target_include_directories(expr_ad_course INTERFACE "${EXPR_AD_INCLUDE_DIR}")

enable_testing()

add_custom_target(expr_ad_all)

function(add_expr_ad_test TARGET SOURCE)
  add_executable(${TARGET} ${SOURCE})
  target_link_libraries(${TARGET} PRIVATE expr_ad_course)
  add_dependencies(expr_ad_all ${TARGET})
  add_test(NAME ${TARGET} COMMAND ${TARGET})
endfunction()

add_expr_ad_test(expr_ad_scaffold tests/test_scaffold.cpp)

if(EXPR_AD_MILESTONE GREATER_EQUAL 1)
  add_expr_ad_test(expr_ad_nodes tests/test_nodes.cpp)
endif()

if(EXPR_AD_MILESTONE GREATER_EQUAL 2)
  add_expr_ad_test(expr_ad_operators tests/test_operators.cpp)
endif()

if(EXPR_AD_MILESTONE GREATER_EQUAL 3)
  add_expr_ad_test(expr_ad_functions tests/test_functions.cpp)
endif()

if(EXPR_AD_MILESTONE GREATER_EQUAL 4)
  add_expr_ad_test(expr_ad_differentiate tests/test_differentiate.cpp)
endif()

if(EXPR_AD_MILESTONE GREATER_EQUAL 5)
  add_expr_ad_test(expr_ad_simplify tests/test_simplify.cpp)
endif()

if(EXPR_AD_MILESTONE GREATER_EQUAL 6)
  add_expr_ad_test(expr_ad_forward_ad tests/test_forward_ad.cpp)
endif()

message(STATUS "ExpressionTemplatesAD milestone ${EXPR_AD_MILESTONE} configured")
