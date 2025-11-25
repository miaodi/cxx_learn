cmake_minimum_required(VERSION 3.18)

# Template Metaprogramming Examples
add_executable(template_tricks
  main.cpp
  always_false_trick.hpp
  type_selection.hpp
  tuple_filtering.hpp
  sfinae_tricks.hpp
)

target_compile_features(template_tricks PRIVATE cxx_std_17)

# Add all source files for better IDE support
target_sources(template_tricks PRIVATE
  always_false_trick.hpp
  type_selection.hpp
  tuple_filtering.hpp
  sfinae_tricks.hpp
)

# # Set output directory
# set_target_properties(template_tricks PROPERTIES
#   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
# )

message(STATUS "Template Metaprogramming examples configured")
