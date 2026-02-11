#pragma once
#include <cstddef>

namespace PMPP {
void merge(int *input1, size_t size1, int *input2, size_t size2, int *output);
void merge_shared(int *input1, size_t size1, int *input2, size_t size2, int *output);
void merge_shared_partitioned(int *input1, size_t size1, int *input2, size_t size2, int *output);
void merge_device(int *d_input1, size_t size1, int *d_input2, size_t size2, int *d_output);
void merge_shared_device(int *d_input1, size_t size1, int *d_input2, size_t size2, int *d_output);
void merge_shared_partitioned_device(int *d_input1, size_t size1, int *d_input2, size_t size2,
                                    int *d_output);
}