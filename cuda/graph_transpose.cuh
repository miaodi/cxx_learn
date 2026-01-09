#pragma once

#include <cuda_runtime.h>

namespace graph_transpose {

// Transpose a CSR graph on CPU.
// Input: row_offsets (num_src_vertices + 1), col_indices (num_edges, dst in [0, num_dst_vertices)).
// Output: out_row_offsets (num_dst_vertices + 1), out_col_indices (num_edges).
// Output edge order within each row is sorted (ascending col index).
void transpose_csr_cpu(const int* row_offsets,
                        const int* col_indices,
                        int num_src_vertices,
                        int num_dst_vertices,
                        int num_edges,
                        int* out_row_offsets,
                        int* out_col_indices);

// Transpose a CSR graph on GPU (device pointers).
// Output edge order within each row is sorted (ascending col index).
// Output buffers must be allocated: out_row_offsets (num_dst_vertices + 1),
// out_col_indices (num_edges).
// If use_coo is false, avoid COO expansion and scatter directly from CSR.
cudaError_t transpose_csr_gpu(const int* d_row_offsets,
                              const int* d_col_indices,
                              int num_src_vertices,
                              int num_dst_vertices,
                              int num_edges,
                              int* d_out_row_offsets,
                              int* d_out_col_indices,
                              bool use_coo = true,
                              cudaStream_t stream = 0);

// Transpose a CSR graph on GPU using global sort on (dst, src).
// Output edge order within each row is sorted (ascending col index).
cudaError_t transpose_csr_gpu_global_sort(const int* d_row_offsets,
                                          const int* d_col_indices,
                                          int num_src_vertices,
                                          int num_dst_vertices,
                                          int num_edges,
                                          int* d_out_row_offsets,
                                          int* d_out_col_indices,
                                          cudaStream_t stream = 0);

}  // namespace graph_transpose
