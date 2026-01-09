#include "graph_transpose.cuh"
#include "mtx_reader.h"

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

mtx_reader::CsrGraph g_graph;
bool g_graph_loaded = false;
std::string g_graph_error;

bool ensure_graph_loaded(benchmark::State& state) {
    if (!g_graph_loaded) {
        std::string message = "Graph not loaded. Provide --mtx=PATH or MTX_PATH.";
        if (!g_graph_error.empty()) {
            message += " Last error: " + g_graph_error;
        }
        state.SkipWithError(message.c_str());
        return false;
    }
    return true;
}

void set_common_metrics(benchmark::State& state) {
    state.counters["src_vertices"] = static_cast<double>(g_graph.num_rows);
    state.counters["dst_vertices"] = static_cast<double>(g_graph.num_cols);
    state.counters["edges"] = static_cast<double>(g_graph.num_edges);
    int64_t bytes_per_iter =
        (static_cast<int64_t>(g_graph.num_rows + g_graph.num_cols + 2) +
         static_cast<int64_t>(2) * g_graph.num_edges) *
        sizeof(int);
    state.SetBytesProcessed(bytes_per_iter * state.iterations());
    state.SetItemsProcessed(static_cast<int64_t>(g_graph.num_edges) *
                            state.iterations());
}

}  // namespace

static void BM_GraphTranspose_CPU(benchmark::State& state) {
    if (!ensure_graph_loaded(state)) {
        return;
    }
    std::vector<int> out_row_offsets(g_graph.num_cols + 1);
    std::vector<int> out_col_indices(g_graph.num_edges);

    for (auto _ : state) {
        graph_transpose::transpose_csr_cpu(
            g_graph.row_offsets.data(),
            g_graph.col_indices.data(),
            g_graph.num_rows,
            g_graph.num_cols,
            g_graph.num_edges,
            out_row_offsets.data(),
            out_col_indices.data());
        benchmark::DoNotOptimize(out_col_indices.data());
        benchmark::ClobberMemory();
    }

    set_common_metrics(state);
}

static void BM_GraphTranspose_GPU(benchmark::State& state, bool use_coo) {
    if (!ensure_graph_loaded(state)) {
        return;
    }

    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
    int* d_out_row_offsets = nullptr;
    int* d_out_col_indices = nullptr;

    cudaError_t status = cudaMalloc(&d_row_offsets,
                                    (g_graph.num_rows + 1) * sizeof(int));
    if (status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(status));
        return;
    }
    status = cudaMalloc(&d_col_indices, g_graph.num_edges * sizeof(int));
    if (status != cudaSuccess) {
        cudaFree(d_row_offsets);
        state.SkipWithError(cudaGetErrorString(status));
        return;
    }
    status = cudaMalloc(&d_out_row_offsets,
                        (g_graph.num_cols + 1) * sizeof(int));
    if (status != cudaSuccess) {
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        state.SkipWithError(cudaGetErrorString(status));
        return;
    }
    status = cudaMalloc(&d_out_col_indices, g_graph.num_edges * sizeof(int));
    if (status != cudaSuccess) {
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_out_row_offsets);
        state.SkipWithError(cudaGetErrorString(status));
        return;
    }

    status = cudaMemcpy(d_row_offsets,
                        g_graph.row_offsets.data(),
                        (g_graph.num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(status));
        goto gpu_cleanup;
    }

    status = cudaMemcpy(d_col_indices,
                        g_graph.col_indices.data(),
                        g_graph.num_edges * sizeof(int),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(status));
        goto gpu_cleanup;
    }

    for (auto _ : state) {
        status = graph_transpose::transpose_csr_gpu(
            d_row_offsets,
            d_col_indices,
            g_graph.num_rows,
            g_graph.num_cols,
            g_graph.num_edges,
            d_out_row_offsets,
            d_out_col_indices,
            use_coo);
        if (status != cudaSuccess) {
            state.SkipWithError(cudaGetErrorString(status));
            break;
        }
    }

    set_common_metrics(state);

gpu_cleanup:
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_out_row_offsets);
    cudaFree(d_out_col_indices);
}

static void BM_GraphTranspose_GPU_GlobalSort(benchmark::State& state) {
    if (!ensure_graph_loaded(state)) {
        return;
    }

    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
    int* d_out_row_offsets = nullptr;
    int* d_out_col_indices = nullptr;

    cudaError_t status = cudaMalloc(&d_row_offsets,
                                    (g_graph.num_rows + 1) * sizeof(int));
    if (status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(status));
        return;
    }
    status = cudaMalloc(&d_col_indices, g_graph.num_edges * sizeof(int));
    if (status != cudaSuccess) {
        cudaFree(d_row_offsets);
        state.SkipWithError(cudaGetErrorString(status));
        return;
    }
    status = cudaMalloc(&d_out_row_offsets,
                        (g_graph.num_cols + 1) * sizeof(int));
    if (status != cudaSuccess) {
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        state.SkipWithError(cudaGetErrorString(status));
        return;
    }
    status = cudaMalloc(&d_out_col_indices, g_graph.num_edges * sizeof(int));
    if (status != cudaSuccess) {
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_out_row_offsets);
        state.SkipWithError(cudaGetErrorString(status));
        return;
    }

    status = cudaMemcpy(d_row_offsets,
                        g_graph.row_offsets.data(),
                        (g_graph.num_rows + 1) * sizeof(int),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(status));
        goto gpu_cleanup;
    }

    status = cudaMemcpy(d_col_indices,
                        g_graph.col_indices.data(),
                        g_graph.num_edges * sizeof(int),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        state.SkipWithError(cudaGetErrorString(status));
        goto gpu_cleanup;
    }

    for (auto _ : state) {
        status = graph_transpose::transpose_csr_gpu_global_sort(
            d_row_offsets,
            d_col_indices,
            g_graph.num_rows,
            g_graph.num_cols,
            g_graph.num_edges,
            d_out_row_offsets,
            d_out_col_indices);
        if (status != cudaSuccess) {
            state.SkipWithError(cudaGetErrorString(status));
            break;
        }
    }

    set_common_metrics(state);

gpu_cleanup:
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_out_row_offsets);
    cudaFree(d_out_col_indices);
}

static void BM_GraphTranspose_GPU_COO(benchmark::State& state) {
    BM_GraphTranspose_GPU(state, true);
}

static void BM_GraphTranspose_GPU_NoCOO(benchmark::State& state) {
    BM_GraphTranspose_GPU(state, false);
}

BENCHMARK(BM_GraphTranspose_CPU)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_GraphTranspose_GPU_COO)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_GraphTranspose_GPU_NoCOO)->Unit(benchmark::kMillisecond)->UseRealTime();
BENCHMARK(BM_GraphTranspose_GPU_GlobalSort)->Unit(benchmark::kMillisecond)->UseRealTime();

int main(int argc, char** argv) {
    std::string mtx_path;
    std::vector<char*> benchmark_args;
    benchmark_args.reserve(static_cast<size_t>(argc));
    benchmark_args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--mtx=", 0) == 0) {
            mtx_path = arg.substr(6);
            continue;
        }
        if (arg == "--mtx" && i + 1 < argc) {
            mtx_path = argv[++i];
            continue;
        }
        benchmark_args.push_back(argv[i]);
    }

    if (mtx_path.empty()) {
        const char* env_path = std::getenv("MTX_PATH");
        if (env_path) {
            mtx_path = env_path;
        }
    }

    if (!mtx_path.empty()) {
        g_graph_loaded = mtx_reader::read_mtx_as_csr(
            mtx_path, &g_graph, &g_graph_error, true);
        if (!g_graph_loaded && g_graph_error.empty()) {
            g_graph_error = "Failed to load mtx file";
        }
    }

    int benchmark_argc = static_cast<int>(benchmark_args.size());
    benchmark::Initialize(&benchmark_argc, benchmark_args.data());
    if (benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_args.data())) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
