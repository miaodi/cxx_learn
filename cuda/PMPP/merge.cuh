#pragma once

namespace PMPP {
void merge(int *input1, int size1, int *input2, int size2, int *output);
void merge_shared(int *input1, int size1, int *input2, int size2, int *output);
void merge_shared_partitioned(int *input1, int size1, int *input2, int size2, int *output);
void merge_device(int *d_input1, int size1, int *d_input2, int size2, int *d_output);
void merge_shared_device(int *d_input1, int size1, int *d_input2, int size2, int *d_output);
void merge_shared_partitioned_device(int *d_input1, int size1, int *d_input2, int size2,
                                    int *d_output);
}