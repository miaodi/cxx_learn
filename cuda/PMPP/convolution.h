#pragma once

// CPU reference implementation
void convolution_2d(const float *input, float *output, const float *kernel,
                    int width, int height, int kernel_size);

// GPU implementations (global memory, constant memory, constant+shared memory)
void convolution_2d_gpu(const float *input, float *output, const float *kernel,
                        int width, int height, int kernel_size);
void convolution_2d_gpu_constmem(const float *input, float *output, const float *kernel,
                                 int width, int height, int kernel_size);
void convolution_2d_gpu_const_shared(const float *input, float *output, const float *kernel,
                                     int width, int height, int kernel_size);