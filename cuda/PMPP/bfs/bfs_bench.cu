#include "bfs.h"
#include "mtx_reader.h"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char *operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::string default_mtx_path() {
  if (const char *path = std::getenv("PMPP_BFS_MTX")) {
    return path;
  }
  return std::string(PMPP_BFS_SAMPLE_DIR) + "/tiny_symmetric.mtx";
}

std::vector<int> make_edge_src_indices(const mtx_reader::CsrGraph &graph) {
  std::vector<int> edge_src_indices;
  edge_src_indices.reserve(graph.num_edges);
  for (int vertex = 0; vertex < graph.num_rows; ++vertex) {
    for (int edge_idx = graph.row_offsets[vertex];
         edge_idx < graph.row_offsets[vertex + 1]; ++edge_idx) {
      edge_src_indices.push_back(vertex);
    }
  }
  return edge_src_indices;
}

class BfsBenchmarkInput {
public:
  BfsBenchmarkInput(std::string path, int source_vertex)
      : source(source_vertex) {
    std::string error;
    if (!mtx_reader::read_mtx_as_csr(path, &graph, &error)) {
      throw std::runtime_error(error);
    }
    if (source < 0 || source >= graph.num_rows) {
      throw std::runtime_error("BFS source vertex is out of range");
    }

    std::cout << "Loaded MTX graph: path=" << path
              << " rows=" << graph.num_rows << " cols=" << graph.num_cols
              << " csr_edges=" << graph.num_edges
              << " source=" << source << '\n';

    check_cuda(cudaMalloc(&d_row_offsets,
                          graph.row_offsets.size() * sizeof(int)),
               "cudaMalloc row_offsets");
    check_cuda(cudaMalloc(&d_col_indices,
                          graph.col_indices.size() * sizeof(int)),
               "cudaMalloc col_indices");
    const std::vector<int> edge_src_indices = make_edge_src_indices(graph);
    check_cuda(cudaMalloc(&d_edge_src_indices,
                          edge_src_indices.size() * sizeof(int)),
               "cudaMalloc edge_src_indices");
    check_cuda(cudaMalloc(&d_levels, graph.num_rows * sizeof(int)),
               "cudaMalloc levels");

    check_cuda(cudaMemcpy(d_row_offsets, graph.row_offsets.data(),
                          graph.row_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy row_offsets");
    check_cuda(cudaMemcpy(d_col_indices, graph.col_indices.data(),
                          graph.col_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy col_indices");
    check_cuda(cudaMemcpy(d_edge_src_indices, edge_src_indices.data(),
                          edge_src_indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy edge_src_indices");
  }

  BfsBenchmarkInput(const BfsBenchmarkInput &) = delete;
  BfsBenchmarkInput &operator=(const BfsBenchmarkInput &) = delete;

  ~BfsBenchmarkInput() {
    cudaFree(d_levels);
    cudaFree(d_edge_src_indices);
    cudaFree(d_col_indices);
    cudaFree(d_row_offsets);
  }

  mtx_reader::CsrGraph graph;
  int source = 0;
  int *d_row_offsets = nullptr;
  int *d_col_indices = nullptr;
  int *d_edge_src_indices = nullptr;
  int *d_levels = nullptr;
};

BfsBenchmarkInput &benchmark_input() {
  static BfsBenchmarkInput input(default_mtx_path(), 0);
  return input;
}

using CsrBfsFunction = cudaError_t (*)(const int *, const int *, int, int,
                                       int *, cudaStream_t);
using EdgeBfsFunction = cudaError_t (*)(const int *, const int *, int, int, int,
                                        int *, cudaStream_t);

void set_graph_counters(benchmark::State &state,
                        const BfsBenchmarkInput &input) {
  state.counters["vertices"] = input.graph.num_rows;
  state.counters["edges"] = input.graph.num_edges;
}

void run_csr_bfs_benchmark(benchmark::State &state,
                           CsrBfsFunction bfs_function) {
  try {
    BfsBenchmarkInput &input = benchmark_input();

    for (auto _ : state) {
      benchmark::DoNotOptimize(input.d_levels);
      const cudaError_t status = bfs_function(
          input.d_row_offsets, input.d_col_indices, input.graph.num_rows,
          input.source, input.d_levels, nullptr);
      if (status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(status));
        break;
      }

      const cudaError_t sync_status = cudaDeviceSynchronize();
      if (sync_status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(sync_status));
        break;
      }
    }

    set_graph_counters(state, input);
  } catch (const std::exception &ex) {
    state.SkipWithError(ex.what());
  }
}

void run_edge_bfs_benchmark(benchmark::State &state,
                            EdgeBfsFunction bfs_function) {
  try {
    BfsBenchmarkInput &input = benchmark_input();

    for (auto _ : state) {
      benchmark::DoNotOptimize(input.d_levels);
      benchmark::DoNotOptimize(input.d_edge_src_indices);
      const cudaError_t status = bfs_function(
          input.d_edge_src_indices, input.d_col_indices, input.graph.num_rows,
          input.graph.num_edges, input.source, input.d_levels, nullptr);
      if (status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(status));
        break;
      }

      const cudaError_t sync_status = cudaDeviceSynchronize();
      if (sync_status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(sync_status));
        break;
      }
    }

    set_graph_counters(state, input);
  } catch (const std::exception &ex) {
    state.SkipWithError(ex.what());
  }
}

void BM_VertexCentricTopDownBfs(benchmark::State &state) {
  run_csr_bfs_benchmark(state, pmpp::bfs::bfs_vertex_centric_top_down);
}

void BM_VertexCentricBottomUpBfs(benchmark::State &state) {
  run_csr_bfs_benchmark(state, pmpp::bfs::bfs_vertex_centric_bottom_up);
}

void BM_EdgeCentricBfs(benchmark::State &state) {
  run_edge_bfs_benchmark(state, pmpp::bfs::bfs_edge_centric);
}

void BM_FrontierTopDownBfs(benchmark::State &state) {
  run_csr_bfs_benchmark(state, pmpp::bfs::bfs_frontier_top_down);
}

BENCHMARK(BM_VertexCentricTopDownBfs)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_VertexCentricBottomUpBfs)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_EdgeCentricBfs)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_FrontierTopDownBfs)->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
