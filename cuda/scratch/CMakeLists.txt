cmake_minimum_required(VERSION 3.18)

if(USE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "native")
    
    # Thrust practice executable
    add_executable(thrust_practice thrust_practice.cu)
    target_compile_features(thrust_practice PRIVATE cxx_std_17)
    target_link_libraries(thrust_practice PRIVATE CUDA::cudart)
    
    # Set CUDA flags for better debugging
    set_target_properties(thrust_practice PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif(USE_CUDA)
