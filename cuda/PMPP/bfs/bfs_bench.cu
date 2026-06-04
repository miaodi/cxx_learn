#include "bfs.h"
#include "mtx_reader.h"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

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
  }

  BfsBenchmarkInput(const BfsBenchmarkInput &) = delete;
  BfsBenchmarkInput &operator=(const BfsBenchmarkInput &) = delete;

  ~BfsBenchmarkInput() {
    cudaFree(d_levels);
    cudaFree(d_col_indices);
    cudaFree(d_row_offsets);
  }

  mtx_reader::CsrGraph graph;
  int source = 0;
  int *d_row_offsets = nullptr;
  int *d_col_indices = nullptr;
  int *d_levels = nullptr;
};

BfsBenchmarkInput &benchmark_input() {
  static BfsBenchmarkInput input(default_mtx_path(), 0);
  return input;
}

void BM_VertexCentricTopDownBfs(benchmark::State &state) {
  try {
    BfsBenchmarkInput &input = benchmark_input();

    for (auto _ : state) {
      benchmark::DoNotOptimize(input.d_levels);
      const cudaError_t status = pmpp::bfs::bfs_vertex_centric_top_down(
          input.d_row_offsets, input.d_col_indices, input.graph.num_rows,
          input.source, input.d_levels);
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

    state.counters["vertices"] = input.graph.num_rows;
    state.counters["edges"] = input.graph.num_edges;
  } catch (const std::exception &ex) {
    state.SkipWithError(ex.what());
  }
}

BENCHMARK(BM_VertexCentricTopDownBfs)->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
