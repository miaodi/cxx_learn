#pragma once

namespace PMPP {
    void merge(int* input1, int size1, int* input2, int size2, int* output);
    void merge_shared(int* input1, int size1, int* input2, int size2, int* output);
}