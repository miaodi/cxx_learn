#include "graph_transpose.cuh"

#include <cub/cub.cuh>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace graph_transpose {
namespace {

constexpr int kBlockSize = 256;

#define CUDA_RETURN_IF_ERROR(call)          \
    do {                                    \
        cudaError_t err = (call);           \
        if (err != cudaSuccess) {           \
            return err;                     \
        }                                   \
    } while (0)

__global__ void count_in_degrees_kernel(const int* col_indices,
                                        int num_edges,
                                        int* out_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int edge = idx; edge < num_edges; edge += stride) {
        int dst = col_indices[edge];
        atomicAdd(&out_counts[dst], 1);
    }
}

__global__ void expand_row_indices_kernel(const int* row_offsets,
                                          int num_vertices,
                                          int* out_src_indices) {
    for (int row = blockIdx.x; row < num_vertices; row += gridDim.x) {
        int start = row_offsets[row];
        int end = row_offsets[row + 1];
        for (int edge = start + threadIdx.x; edge < end; edge += blockDim.x) {
            out_src_indices[edge] = row;
        }
    }
}

__global__ void scatter_transpose_kernel(const int* src_indices,
                                         const int* dst_indices,
                                         int num_edges,
                                         int* out_positions,
                                         int* out_col_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int edge = idx; edge < num_edges; edge += stride) {
        int dst = dst_indices[edge];
        int pos = atomicAdd(&out_positions[dst], 1);
        out_col_indices[pos] = src_indices[edge];
    }
}

__global__ void scatter_transpose_no_coo_kernel(const int* row_offsets,
                                                const int* col_indices,
                                                int num_src_vertices,
                                                int* out_positions,
                                                int* out_col_indices) {
    for (int row = blockIdx.x; row < num_src_vertices; row += gridDim.x) {
        int start = row_offsets[row];
        int end = row_offsets[row + 1];
        for (int edge = start + threadIdx.x; edge < end; edge += blockDim.x) {
            int dst = col_indices[edge];
            int pos = atomicAdd(&out_positions[dst], 1);
            out_col_indices[pos] = row;
        }
    }
}

__global__ void build_keys_from_csr_kernel(const int* row_offsets,
                                           const int* col_indices,
                                           int num_src_vertices,
                                           int shift_bits,
                                           uint64_t* out_keys) {
    for (int row = blockIdx.x; row < num_src_vertices; row += gridDim.x) {
        int start = row_offsets[row];
        int end = row_offsets[row + 1];
        uint64_t src = static_cast<uint64_t>(static_cast<uint32_t>(row));
        for (int edge = start + threadIdx.x; edge < end; edge += blockDim.x) {
            uint64_t dst = static_cast<uint64_t>(
                static_cast<uint32_t>(col_indices[edge]));
            out_keys[edge] = (dst << shift_bits) | src;
        }
    }
}

__global__ void extract_src_from_keys_kernel(const uint64_t* keys,
                                             int num_edges,
                                             uint64_t src_mask,
                                             int* out_col_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int edge = idx; edge < num_edges; edge += stride) {
        out_col_indices[edge] = static_cast<int>(keys[edge] & src_mask);
    }
}

int clamp_blocks(int blocks) {
    const int max_blocks = 65535;
    return blocks > max_blocks ? max_blocks : blocks;
}

int bits_required(int value) {
    int bits = 0;
    while (value > 0) {
        ++bits;
        value >>= 1;
    }
    return bits;
}

}  // namespace

void transpose_csr_cpu(const int* row_offsets,
                        const int* col_indices,
                        int num_src_vertices,
                        int num_dst_vertices,
                        int num_edges,
                        int* out_row_offsets,
                        int* out_col_indices) {
    std::fill(out_row_offsets, out_row_offsets + num_dst_vertices + 1, 0);
    for (int edge = 0; edge < num_edges; ++edge) {
        int dst = col_indices[edge];
        ++out_row_offsets[dst + 1];
    }

    for (int v = 0; v < num_dst_vertices; ++v) {
        out_row_offsets[v + 1] += out_row_offsets[v];
    }

    std::vector<int> positions(out_row_offsets, out_row_offsets + num_dst_vertices);
    for (int src = 0; src < num_src_vertices; ++src) {
        int start = row_offsets[src];
        int end = row_offsets[src + 1];
        for (int edge = start; edge < end; ++edge) {
            int dst = col_indices[edge];
            int pos = positions[dst]++;
            out_col_indices[pos] = src;
        }
    }

    for (int row = 0; row < num_dst_vertices; ++row) {
        int start = out_row_offsets[row];
        int end = out_row_offsets[row + 1];
        std::sort(out_col_indices + start, out_col_indices + end);
    }
}

cudaError_t transpose_csr_gpu(const int* d_row_offsets,
                              const int* d_col_indices,
                              int num_src_vertices,
                              int num_dst_vertices,
                              int num_edges,
                              int* d_out_row_offsets,
                              int* d_out_col_indices,
                              bool use_coo,
                              cudaStream_t stream) {
    if (num_src_vertices < 0 || num_dst_vertices < 0 || num_edges < 0) {
        return cudaErrorInvalidValue;
    }
    if (num_src_vertices == 0 && num_edges > 0) {
        return cudaErrorInvalidValue;
    }
    if (num_dst_vertices == 0) {
        if (num_edges > 0) {
            return cudaErrorInvalidValue;
        }
        CUDA_RETURN_IF_ERROR(
            cudaMemsetAsync(d_out_row_offsets, 0, sizeof(int), stream));
        return cudaSuccess;
    }
    if (num_edges == 0) {
        CUDA_RETURN_IF_ERROR(
            cudaMemsetAsync(d_out_row_offsets, 0,
                            (num_dst_vertices + 1) * sizeof(int), stream));
        return cudaSuccess;
    }

    int* d_counts = nullptr;
    int* d_src_indices = nullptr;
    int* d_tmp_col_indices = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    size_t sort_temp_storage_bytes = 0;
    int edge_blocks = 0;
    int row_blocks = 0;

    cudaError_t status = cudaSuccess;
    status = cudaMalloc(&d_counts, num_dst_vertices * sizeof(int));
    if (status != cudaSuccess) {
        goto cleanup;
    }
    if (use_coo) {
        status = cudaMalloc(&d_src_indices, num_edges * sizeof(int));
        if (status != cudaSuccess) {
            goto cleanup;
        }
    }
    status = cudaMemsetAsync(d_counts, 0, num_dst_vertices * sizeof(int), stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }
    status = cudaMemsetAsync(d_out_row_offsets, 0, sizeof(int), stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }

    edge_blocks = clamp_blocks((num_edges + kBlockSize - 1) / kBlockSize);
    count_in_degrees_kernel<<<edge_blocks, kBlockSize, 0, stream>>>(
        d_col_indices, num_edges, d_counts);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        goto cleanup;
    }

    status = cudaMalloc(&d_tmp_col_indices, num_edges * sizeof(int));
    if (status != cudaSuccess) {
        goto cleanup;
    }

    // Leave row_offsets[0] at 0, scan into row_offsets[1..].
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_counts, d_out_row_offsets + 1,
                                  num_dst_vertices, stream);
    status = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (status != cudaSuccess) {
        goto cleanup;
    }
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_counts, d_out_row_offsets + 1,
                                  num_dst_vertices, stream);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        goto cleanup;
    }

    // Covert to COO format: expand row indices.
    row_blocks = clamp_blocks(num_src_vertices);
    if (use_coo) {
        // Convert to COO format: expand row indices.
        expand_row_indices_kernel<<<row_blocks, kBlockSize, 0, stream>>>(
            d_row_offsets, num_src_vertices, d_src_indices);
        status = cudaGetLastError();
        if (status != cudaSuccess) {
            goto cleanup;
        }

        // Use row_offsets[1..] as positions; atomics advance to final offsets.
        scatter_transpose_kernel<<<edge_blocks, kBlockSize, 0, stream>>>(
            d_src_indices, d_col_indices, num_edges,
            d_out_row_offsets + 1, d_tmp_col_indices);
    } else {
        // Scatter directly from CSR with worse load balance but less memory.
        scatter_transpose_no_coo_kernel<<<row_blocks, kBlockSize, 0, stream>>>(
            d_row_offsets, d_col_indices, num_src_vertices,
            d_out_row_offsets + 1, d_tmp_col_indices);
    }
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        goto cleanup;
    }

    // Sort each row segment to produce ordered adjacency lists.
    sort_temp_storage_bytes = 0;
    status = cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, sort_temp_storage_bytes,
        d_tmp_col_indices, d_out_col_indices,
        num_edges, num_dst_vertices,
        d_out_row_offsets, d_out_row_offsets + 1,
        0, static_cast<int>(sizeof(int) * 8), stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }
    if (sort_temp_storage_bytes > temp_storage_bytes) {
        cudaFree(d_temp_storage);
        d_temp_storage = nullptr;
        temp_storage_bytes = sort_temp_storage_bytes;
        status = cudaMalloc(&d_temp_storage, temp_storage_bytes);
        if (status != cudaSuccess) {
            goto cleanup;
        }
    }
    status = cub::DeviceSegmentedRadixSort::SortKeys(
        d_temp_storage, sort_temp_storage_bytes,
        d_tmp_col_indices, d_out_col_indices,
        num_edges, num_dst_vertices,
        d_out_row_offsets, d_out_row_offsets + 1,
        0, static_cast<int>(sizeof(int) * 8), stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }

    status = cudaStreamSynchronize(stream);

cleanup:
    cudaFree(d_counts);
    cudaFree(d_src_indices);
    cudaFree(d_tmp_col_indices);
    cudaFree(d_temp_storage);

    return status;
}

cudaError_t transpose_csr_gpu_global_sort(const int* d_row_offsets,
                                          const int* d_col_indices,
                                          int num_src_vertices,
                                          int num_dst_vertices,
                                          int num_edges,
                                          int* d_out_row_offsets,
                                          int* d_out_col_indices,
                                          cudaStream_t stream) {
    if (num_src_vertices < 0 || num_dst_vertices < 0 || num_edges < 0) {
        return cudaErrorInvalidValue;
    }
    if (num_src_vertices == 0 && num_edges > 0) {
        return cudaErrorInvalidValue;
    }
    if (num_dst_vertices == 0) {
        if (num_edges > 0) {
            return cudaErrorInvalidValue;
        }
        CUDA_RETURN_IF_ERROR(
            cudaMemsetAsync(d_out_row_offsets, 0, sizeof(int), stream));
        return cudaSuccess;
    }
    if (num_edges == 0) {
        CUDA_RETURN_IF_ERROR(
            cudaMemsetAsync(d_out_row_offsets, 0,
                            (num_dst_vertices + 1) * sizeof(int), stream));
        return cudaSuccess;
    }

    int bits_src = bits_required(num_src_vertices - 1);
    int bits_dst = bits_required(num_dst_vertices - 1);
    if (bits_src > 32) {
        return cudaErrorInvalidValue;
    }
    int end_bit = bits_src + bits_dst;
    if (end_bit == 0) {
        end_bit = 1;
    }
    uint64_t src_mask = (bits_src == 0)
        ? 0ULL
        : ((bits_src == 64) ? ~0ULL : ((1ULL << bits_src) - 1ULL));

    int* d_counts = nullptr;
    uint64_t* d_keys = nullptr;
    uint64_t* d_keys_sorted = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    size_t sort_temp_storage_bytes = 0;
    int edge_blocks = 0;
    int row_blocks = 0;

    cudaError_t status = cudaSuccess;
    status = cudaMalloc(&d_counts, num_dst_vertices * sizeof(int));
    if (status != cudaSuccess) {
        goto cleanup;
    }
    status = cudaMalloc(&d_keys, num_edges * sizeof(uint64_t));
    if (status != cudaSuccess) {
        goto cleanup;
    }
    status = cudaMalloc(&d_keys_sorted, num_edges * sizeof(uint64_t));
    if (status != cudaSuccess) {
        goto cleanup;
    }

    status = cudaMemsetAsync(d_counts, 0, num_dst_vertices * sizeof(int), stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }
    status = cudaMemsetAsync(d_out_row_offsets, 0, sizeof(int), stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }

    edge_blocks = clamp_blocks((num_edges + kBlockSize - 1) / kBlockSize);
    count_in_degrees_kernel<<<edge_blocks, kBlockSize, 0, stream>>>(
        d_col_indices, num_edges, d_counts);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        goto cleanup;
    }

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_counts, d_out_row_offsets + 1,
                                  num_dst_vertices, stream);
    status = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (status != cudaSuccess) {
        goto cleanup;
    }
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_counts, d_out_row_offsets + 1,
                                  num_dst_vertices, stream);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        goto cleanup;
    }

    row_blocks = clamp_blocks(num_src_vertices);
    build_keys_from_csr_kernel<<<row_blocks, kBlockSize, 0, stream>>>(
        d_row_offsets, d_col_indices, num_src_vertices, bits_src, d_keys);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        goto cleanup;
    }

    sort_temp_storage_bytes = 0;
    status = cub::DeviceRadixSort::SortKeys(
        nullptr, sort_temp_storage_bytes,
        d_keys, d_keys_sorted,
        num_edges, 0, end_bit, stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }
    if (sort_temp_storage_bytes > temp_storage_bytes) {
        cudaFree(d_temp_storage);
        d_temp_storage = nullptr;
        temp_storage_bytes = sort_temp_storage_bytes;
        status = cudaMalloc(&d_temp_storage, temp_storage_bytes);
        if (status != cudaSuccess) {
            goto cleanup;
        }
    }
    status = cub::DeviceRadixSort::SortKeys(
        d_temp_storage, sort_temp_storage_bytes,
        d_keys, d_keys_sorted,
        num_edges, 0, end_bit, stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }

    extract_src_from_keys_kernel<<<edge_blocks, kBlockSize, 0, stream>>>(
        d_keys_sorted, num_edges, src_mask, d_out_col_indices);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        goto cleanup;
    }

    status = cudaStreamSynchronize(stream);

cleanup:
    cudaFree(d_counts);
    cudaFree(d_keys);
    cudaFree(d_keys_sorted);
    cudaFree(d_temp_storage);

    return status;
}

cudaError_t transpose_csr_gpu_kv_sort(const int* d_row_offsets,
                                      const int* d_col_indices,
                                      int num_src_vertices,
                                      int num_dst_vertices,
                                      int num_edges,
                                      int* d_out_row_offsets,
                                      int* d_out_col_indices,
                                      cudaStream_t stream) {
    if (num_src_vertices < 0 || num_dst_vertices < 0 || num_edges < 0) {
        return cudaErrorInvalidValue;
    }
    if (num_src_vertices == 0 && num_edges > 0) {
        return cudaErrorInvalidValue;
    }
    if (num_dst_vertices == 0) {
        if (num_edges > 0) {
            return cudaErrorInvalidValue;
        }
        CUDA_RETURN_IF_ERROR(
            cudaMemsetAsync(d_out_row_offsets, 0, sizeof(int), stream));
        return cudaSuccess;
    }
    if (num_edges == 0) {
        CUDA_RETURN_IF_ERROR(
            cudaMemsetAsync(d_out_row_offsets, 0,
                            (num_dst_vertices + 1) * sizeof(int), stream));
        return cudaSuccess;
    }

    int* d_counts = nullptr;
    int* d_src_indices = nullptr;
    int* d_dst_indices_copy = nullptr;
    int* d_src_sorted = nullptr;
    int* d_dst_sorted = nullptr;
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    size_t sort_temp_storage_bytes = 0;
    int edge_blocks = 0;
    int row_blocks = 0;

    cudaError_t status = cudaSuccess;
    
    // Allocate memory
    status = cudaMalloc(&d_counts, num_dst_vertices * sizeof(int));
    if (status != cudaSuccess) {
        goto cleanup;
    }
    status = cudaMalloc(&d_src_indices, num_edges * sizeof(int));
    if (status != cudaSuccess) {
        goto cleanup;
    }
    status = cudaMalloc(&d_dst_indices_copy, num_edges * sizeof(int));
    if (status != cudaSuccess) {
        goto cleanup;
    }
    status = cudaMalloc(&d_src_sorted, num_edges * sizeof(int));
    if (status != cudaSuccess) {
        goto cleanup;
    }
    status = cudaMalloc(&d_dst_sorted, num_edges * sizeof(int));
    if (status != cudaSuccess) {
        goto cleanup;
    }

    status = cudaMemsetAsync(d_counts, 0, num_dst_vertices * sizeof(int), stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }
    status = cudaMemsetAsync(d_out_row_offsets, 0, sizeof(int), stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }

    // Count in-degrees
    edge_blocks = clamp_blocks((num_edges + kBlockSize - 1) / kBlockSize);
    count_in_degrees_kernel<<<edge_blocks, kBlockSize, 0, stream>>>(
        d_col_indices, num_edges, d_counts);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        goto cleanup;
    }

    // Build row offsets via inclusive scan
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_counts, d_out_row_offsets + 1,
                                  num_dst_vertices, stream);
    status = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (status != cudaSuccess) {
        goto cleanup;
    }
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_counts, d_out_row_offsets + 1,
                                  num_dst_vertices, stream);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        goto cleanup;
    }

    // Expand row indices to COO format (creates source array)
    row_blocks = clamp_blocks(num_src_vertices);
    expand_row_indices_kernel<<<row_blocks, kBlockSize, 0, stream>>>(
        d_row_offsets, num_src_vertices, d_src_indices);
    status = cudaGetLastError();
    if (status != cudaSuccess) {
        goto cleanup;
    }

    // Copy destination indices (will be used as sort keys)
    status = cudaMemcpyAsync(d_dst_indices_copy, d_col_indices,
                             num_edges * sizeof(int),
                             cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }

    // Sort pairs: keys=destinations, values=sources
    // This groups edges by destination and sorts sources within each group
    sort_temp_storage_bytes = 0;
    status = cub::DeviceRadixSort::SortPairs(
        nullptr, sort_temp_storage_bytes,
        d_dst_indices_copy, d_dst_sorted,
        d_src_indices, d_src_sorted,
        num_edges, 0, static_cast<int>(sizeof(int) * 8), stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }
    if (sort_temp_storage_bytes > temp_storage_bytes) {
        cudaFree(d_temp_storage);
        d_temp_storage = nullptr;
        temp_storage_bytes = sort_temp_storage_bytes;
        status = cudaMalloc(&d_temp_storage, temp_storage_bytes);
        if (status != cudaSuccess) {
            goto cleanup;
        }
    }
    status = cub::DeviceRadixSort::SortPairs(
        d_temp_storage, sort_temp_storage_bytes,
        d_dst_indices_copy, d_dst_sorted,
        d_src_indices, d_src_sorted,
        num_edges, 0, static_cast<int>(sizeof(int) * 8), stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }

    // Copy sorted sources to output (these are the new column indices)
    status = cudaMemcpyAsync(d_out_col_indices, d_src_sorted,
                             num_edges * sizeof(int),
                             cudaMemcpyDeviceToDevice, stream);
    if (status != cudaSuccess) {
        goto cleanup;
    }

    status = cudaStreamSynchronize(stream);

cleanup:
    cudaFree(d_counts);
    cudaFree(d_src_indices);
    cudaFree(d_dst_indices_copy);
    cudaFree(d_src_sorted);
    cudaFree(d_dst_sorted);
    cudaFree(d_temp_storage);

    return status;
}

}  // namespace graph_transpose
