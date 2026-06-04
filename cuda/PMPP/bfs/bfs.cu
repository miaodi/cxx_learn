#include "bfs.h"

#include <cuda_runtime.h>

namespace pmpp::bfs {
namespace {

constexpr int kBlockSize = 256;

__global__ void initialize_levels_kernel(int *levels, int num_vertices,
                                         int source) {
  const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= num_vertices) {
    return;
  }

  levels[vertex] = (vertex == source) ? 0 : kUnvisited;
}

__global__ void bfs_vertex_centric_kernel(const int *row_offsets,
                                          const int *col_indices,
                                          int num_vertices,
                                          int current_level,
                                          int *levels,
                                          int *changed) {
  const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= num_vertices) {
    return;
  }
  if (levels[vertex] != current_level) {
    return;
  }
  for(int edge_idx = row_offsets[vertex]; edge_idx < row_offsets[vertex + 1];
      ++edge_idx) {
    const int neighbor = col_indices[edge_idx];
    if (levels[neighbor] == kUnvisited) {
      levels[neighbor] = current_level + 1;
      *changed = 1;
    }
  }
}

} // namespace

cudaError_t bfs_vertex_centric_top_down(const int *d_row_offsets,
                                        const int *d_col_indices,
                                        int num_vertices,
                                        int source,
                                        int *d_levels,
                                        cudaStream_t stream) {
  if (!d_row_offsets || !d_col_indices || !d_levels || num_vertices < 0 ||
      source < 0 || source >= num_vertices) {
    return cudaErrorInvalidValue;
  }

  if (num_vertices == 0) {
    return cudaSuccess;
  }

  const int blocks = (num_vertices + kBlockSize - 1) / kBlockSize;

  initialize_levels_kernel<<<blocks, kBlockSize, 0, stream>>>(
      d_levels, num_vertices, source);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return status;
  }

  int *d_changed = nullptr;
  status = cudaMalloc(&d_changed, sizeof(int));
  if (status != cudaSuccess) {
    return status;
  }

  for (int current_level = 0;; ++current_level) {
    int changed = 0;
    status = cudaMemcpyAsync(d_changed, &changed, sizeof(int),
                             cudaMemcpyHostToDevice, stream);
    if (status != cudaSuccess) {
      cudaFree(d_changed);
      return status;
    }

    bfs_vertex_centric_kernel<<<blocks, kBlockSize, 0, stream>>>(
        d_row_offsets, d_col_indices, num_vertices, current_level, d_levels,
        d_changed);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cudaFree(d_changed);
      return status;
    }

    status = cudaMemcpyAsync(&changed, d_changed, sizeof(int),
                             cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) {
      cudaFree(d_changed);
      return status;
    }

    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      cudaFree(d_changed);
      return status;
    }

    if (changed == 0) {
      break;
    }
  }

  return cudaFree(d_changed);
}

} // namespace pmpp::bfs
