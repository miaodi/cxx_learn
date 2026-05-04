cmake_minimum_required(VERSION 3.5...3.20)
project(Branch CXX)

set(SRC bench.cpp)

if(CXX_LEARNING_HAS_BENCHMARK)
  set(DEPENDENCIES benchmark::benchmark OpenMP::OpenMP_CXX)

  add_executable(${PROJECT_NAME} ${SRC})
  # set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME ${ONE_BENCH_EXEC}) 
  target_link_libraries(${PROJECT_NAME} PRIVATE ${DEPENDENCIES})
endif()
