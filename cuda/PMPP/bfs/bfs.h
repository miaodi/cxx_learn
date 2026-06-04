#pragma once

#include <cuda_runtime_api.h>

namespace pmpp::bfs {

constexpr int kUnvisited = -1;

// PMPP Ch. 15.3 style BFS: each iteration launches over all vertices.
// The current frontier is implicit: levels[v] == current_level.
cudaError_t bfs_vertex_centric_top_down(const int *d_row_offsets,
                                        const int *d_col_indices,
                                        int num_vertices,
                                        int source,
                                        int *d_levels,
                                        cudaStream_t stream = nullptr);

} // namespace pmpp::bfs
