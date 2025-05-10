cmake_minimum_required(VERSION 3.5...3.20)
project(AssemblyStudy CXX)


set(DEPENDENCIES benchmark::benchmark)

add_executable(swap swap.cpp)
target_link_libraries(swap PRIVATE ${DEPENDENCIES})