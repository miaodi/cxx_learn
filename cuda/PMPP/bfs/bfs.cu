#include "bfs.h"

#include <cuda_runtime.h>

#include <utility>

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
                                          int num_vertices, int current_level,
                                          int *levels, int *changed) {
  const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= num_vertices) {
    return;
  }
  if (levels[vertex] != current_level) {
    return;
  }
  for (int adj_idx = row_offsets[vertex]; adj_idx < row_offsets[vertex + 1];
       ++adj_idx) {
    const int adj = col_indices[adj_idx];
    if (levels[adj] == kUnvisited) {
      levels[adj] = current_level + 1;
      *changed = 1;
    }
  }
}

} // namespace

cudaError_t bfs_vertex_centric_top_down(const int *d_row_offsets,
                                        const int *d_col_indices,
                                        int num_vertices, int source,
                                        int *d_levels, cudaStream_t stream) {
  if (num_vertices < 0) {
    return cudaErrorInvalidValue;
  }

  if (num_vertices == 0) {
    return cudaSuccess;
  }

  if (!d_row_offsets || !d_col_indices || !d_levels || source < 0 ||
      source >= num_vertices) {
    return cudaErrorInvalidValue;
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
    status = cudaMemsetAsync(d_changed, 0, sizeof(int), stream);
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
    int h_changed = 0;
    status = cudaMemcpyAsync(&h_changed, d_changed, sizeof(int),
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
    if (h_changed == 0) {
      break;
    }
  }

  return cudaFree(d_changed);
}

__global__ void bfs_vertice_centric_bottom_up_kernel(
    const int *row_offsets, const int *col_indices, int num_vertices,
    int current_level, int *levels, int *changed) {
  const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
  if (vertex >= num_vertices) {
    return;
  }
  for (int adj_idx = row_offsets[vertex]; adj_idx < row_offsets[vertex + 1];
       ++adj_idx) {
    const int adj = col_indices[adj_idx];
    if (levels[adj] == current_level && levels[vertex] == kUnvisited) {
      levels[vertex] = current_level + 1;
      *changed = 1;
      break;
    }
  }
}

cudaError_t bfs_vertex_centric_bottom_up(const int *d_row_offsets,
                                         const int *d_col_indices,
                                         int num_vertices, int source,
                                         int *d_levels, cudaStream_t stream) {
  if (num_vertices < 0) {
    return cudaErrorInvalidValue;
  }
  if (!d_row_offsets || !d_col_indices || !d_levels || source < 0 ||
      source >= num_vertices) {
    return cudaErrorInvalidValue;
  }

  const int blocks = (num_vertices + kBlockSize - 1) / kBlockSize;
  int *d_changed = nullptr;
  cudaError_t status = cudaMalloc(&d_changed, sizeof(int));
  if (status != cudaSuccess) {
    return status;
  }

  initialize_levels_kernel<<<blocks, kBlockSize, 0, stream>>>(
      d_levels, num_vertices, source);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cudaFree(d_changed);
    return status;
  }

  for (int current_level = 0;; ++current_level) {
    status = cudaMemsetAsync(d_changed, 0, sizeof(int), stream);
    if (status != cudaSuccess) {
      cudaFree(d_changed);
      return status;
    }
    bfs_vertice_centric_bottom_up_kernel<<<blocks, kBlockSize, 0, stream>>>(
        d_row_offsets, d_col_indices, num_vertices, current_level, d_levels,
        d_changed);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cudaFree(d_changed);
      return status;
    }
    int h_changed = 0;
    status = cudaMemcpyAsync(&h_changed, d_changed, sizeof(int),
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
    if (h_changed == 0) {
      break;
    }
  }
  cudaFree(d_changed);
  return cudaSuccess;
}

__global__ void bfs_edge_centric_kernel(const int *src_indices,
                                        const int *dst_indices, int num_edges,
                                        int current_level, int *levels,
                                        int *changed) {
  const int edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= num_edges) {
    return;
  }
  const int src = src_indices[edge];
  const int dst = dst_indices[edge];
  if (levels[src] == current_level && levels[dst] == kUnvisited) {
    levels[dst] = current_level + 1;
    *changed = 1;
  }
}

cudaError_t bfs_edge_centric(const int *d_src_indices, const int *d_dst_indices,
                             int num_vertices, int num_edges, int source,
                             int *d_levels, cudaStream_t stream) {
  if (num_vertices < 0 || num_edges < 0) {
    return cudaErrorInvalidValue;
  }
  if (!d_src_indices || !d_dst_indices || !d_levels || source < 0 ||
      source >= num_vertices) {
    return cudaErrorInvalidValue;
  }
  if (num_edges == 0) {
    return cudaSuccess;
  }

  const int blocks = (num_edges + kBlockSize - 1) / kBlockSize;
  int *d_changed = nullptr;
  cudaError_t status = cudaMalloc(&d_changed, sizeof(int));
  if (status != cudaSuccess) {
    return status;
  }

  initialize_levels_kernel<<<(num_vertices + kBlockSize - 1) / kBlockSize,
                             kBlockSize, 0, stream>>>(d_levels, num_vertices,
                                                      source);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cudaFree(d_changed);
    return status;
  }

  for (int current_level = 0;; ++current_level) {
    status = cudaMemsetAsync(d_changed, 0, sizeof(int), stream);
    if (status != cudaSuccess) {
      cudaFree(d_changed);
      return status;
    }
    bfs_edge_centric_kernel<<<blocks, kBlockSize, 0, stream>>>(
        d_src_indices, d_dst_indices, num_edges, current_level, d_levels,
        d_changed);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cudaFree(d_changed);
      return status;
    }
    int h_changed = 0;
    status = cudaMemcpyAsync(&h_changed, d_changed, sizeof(int),
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
    if (h_changed == 0) {
      break;
    }
  }
  cudaFree(d_changed);
  return cudaSuccess;
}

constexpr int kFrontierSize = 1024;

__global__ void bfs_frontier_top_down_kernel(const int *row_offsets,
                                             const int *col_indices,
                                             int *frontier, int frontier_size,
                                             int current_level, int *levels,
                                             int *next_frontier,
                                             int *next_frontier_size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ int shared_frontier[kFrontierSize];
  __shared__ int shared_idx;
  if (threadIdx.x == 0) {
    shared_idx = 0;
  }
  __syncthreads();
  if (idx < frontier_size) {
    const int vertex = frontier[idx];
    for (int adj_idx = row_offsets[vertex]; adj_idx < row_offsets[vertex + 1];
         ++adj_idx) {
      const int adj = col_indices[adj_idx];
      if (atomicCAS(&levels[adj], kUnvisited, current_level + 1) ==
          kUnvisited) {
        int pos = atomicAdd(&shared_idx, 1);
        if (pos < kFrontierSize) {
          shared_frontier[pos] = adj;
        } else {
          pos = atomicAdd(next_frontier_size, 1);
          next_frontier[pos] = adj;
        }
      }
    }
  }

  __syncthreads();
  int shared_count = min(shared_idx, kFrontierSize);
  __shared__ int global_pos;
  if (threadIdx.x == 0) {
    global_pos = atomicAdd(next_frontier_size, shared_count);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < shared_count; i += blockDim.x) {
    next_frontier[global_pos + i] = shared_frontier[i];
  }
}

cudaError_t bfs_frontier_top_down(const int *d_row_offsets,
                                  const int *d_col_indices, int num_vertices,
                                  int source, int *d_levels,
                                  cudaStream_t stream) {
  if (num_vertices < 0) {
    return cudaErrorInvalidValue;
  }

  if (num_vertices == 0) {
    return cudaSuccess;
  }

  if (!d_row_offsets || !d_col_indices || !d_levels || source < 0 ||
      source >= num_vertices) {
    return cudaErrorInvalidValue;
  }

  int *d_frontier = nullptr;
  int *d_next_frontier = nullptr;
  int *d_next_frontier_size = nullptr;

  cudaError_t status = cudaMalloc(&d_frontier, num_vertices * sizeof(int));
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaMalloc(&d_next_frontier, num_vertices * sizeof(int));
  if (status != cudaSuccess) {
    cudaFree(d_frontier);
    return status;
  }
  status = cudaMalloc(&d_next_frontier_size, sizeof(int));
  if (status != cudaSuccess) {
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    return status;
  }

  int h_frontier_size = 1;
  status = cudaMemcpyAsync(d_frontier, &source, sizeof(int),
                           cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) {
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_frontier_size);
    return status;
  }

  initialize_levels_kernel<<<(num_vertices + kBlockSize - 1) / kBlockSize,
                             kBlockSize, 0, stream>>>(d_levels, num_vertices,
                                                       source);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    cudaFree(d_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_next_frontier_size);
    return status;
  }

  for (int level = 0;; ++level) {
    status = cudaMemsetAsync(d_next_frontier_size, 0, sizeof(int), stream);
    if (status != cudaSuccess) {
      cudaFree(d_frontier);
      cudaFree(d_next_frontier);
      cudaFree(d_next_frontier_size);
      return status;
    }
    const int blocks = (h_frontier_size + kBlockSize - 1) / kBlockSize;
    bfs_frontier_top_down_kernel<<<blocks, kBlockSize, 0, stream>>>(
        d_row_offsets, d_col_indices, d_frontier, h_frontier_size, level,
        d_levels, d_next_frontier, d_next_frontier_size);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
      cudaFree(d_frontier);
      cudaFree(d_next_frontier);
      cudaFree(d_next_frontier_size);
      return status;
    }
    status = cudaMemcpyAsync(&h_frontier_size, d_next_frontier_size,
                             sizeof(int), cudaMemcpyDeviceToHost, stream);
    if (status != cudaSuccess) {
      cudaFree(d_frontier);
      cudaFree(d_next_frontier);
      cudaFree(d_next_frontier_size);
      return status;
    }
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      cudaFree(d_frontier);
      cudaFree(d_next_frontier);
      cudaFree(d_next_frontier_size);
      return status;
    }
    if (h_frontier_size == 0) {
      break;
    }
    std::swap(d_frontier, d_next_frontier);
  }

  cudaFree(d_frontier);
  cudaFree(d_next_frontier);
  cudaFree(d_next_frontier_size);
  return cudaSuccess;
}

} // namespace pmpp::bfs
