#include "graph_transpose.cuh"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

namespace {

// Helper to compare CSR graphs
bool compare_csr(const std::vector<int>& row_offsets1,
                 const std::vector<int>& col_indices1,
                 const std::vector<int>& row_offsets2,
                 const std::vector<int>& col_indices2) {
    if (row_offsets1.size() != row_offsets2.size()) {
        return false;
    }
    if (col_indices1.size() != col_indices2.size()) {
        return false;
    }
    
    int num_vertices = row_offsets1.size() - 1;
    for (int v = 0; v < num_vertices; ++v) {
        int start1 = row_offsets1[v];
        int end1 = row_offsets1[v + 1];
        int start2 = row_offsets2[v];
        int end2 = row_offsets2[v + 1];
        
        if (end1 - start1 != end2 - start2) {
            return false;
        }
        
        // Compare sorted adjacency lists
        std::vector<int> adj1(col_indices1.begin() + start1, col_indices1.begin() + end1);
        std::vector<int> adj2(col_indices2.begin() + start2, col_indices2.begin() + end2);
        std::sort(adj1.begin(), adj1.end());
        std::sort(adj2.begin(), adj2.end());
        
        if (adj1 != adj2) {
            return false;
        }
    }
    return true;
}

}  // namespace

class GraphTransposeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Small test graph:
        // 0 -> 1, 2
        // 1 -> 2
        // 2 -> 0
        // 3 -> 1
        row_offsets_h = {0, 2, 3, 4, 5};
        col_indices_h = {1, 2, 2, 0, 1};
        num_src_vertices = 4;
        num_dst_vertices = 3;
        num_edges = 5;
        
        // Expected transpose:
        // 0 -> 2
        // 1 -> 0, 3
        // 2 -> 0, 1
        expected_row_offsets = {0, 1, 3, 5};
        expected_col_indices = {2, 0, 3, 0, 1};
        
        // Allocate device memory
        cudaMalloc(&d_row_offsets, (num_src_vertices + 1) * sizeof(int));
        cudaMalloc(&d_col_indices, num_edges * sizeof(int));
        cudaMalloc(&d_out_row_offsets, (num_dst_vertices + 1) * sizeof(int));
        cudaMalloc(&d_out_col_indices, num_edges * sizeof(int));
        
        // Copy input to device
        cudaMemcpy(d_row_offsets, row_offsets_h.data(),
                   (num_src_vertices + 1) * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_indices, col_indices_h.data(),
                   num_edges * sizeof(int),
                   cudaMemcpyHostToDevice);
    }
    
    void TearDown() override {
        cudaFree(d_row_offsets);
        cudaFree(d_col_indices);
        cudaFree(d_out_row_offsets);
        cudaFree(d_out_col_indices);
    }
    
    std::vector<int> row_offsets_h;
    std::vector<int> col_indices_h;
    std::vector<int> expected_row_offsets;
    std::vector<int> expected_col_indices;
    int num_src_vertices;
    int num_dst_vertices;
    int num_edges;
    
    int* d_row_offsets = nullptr;
    int* d_col_indices = nullptr;
    int* d_out_row_offsets = nullptr;
    int* d_out_col_indices = nullptr;
};

TEST_F(GraphTransposeTest, CPU_Transpose) {
    std::vector<int> out_row_offsets(num_dst_vertices + 1);
    std::vector<int> out_col_indices(num_edges);
    
    graph_transpose::transpose_csr_cpu(
        row_offsets_h.data(), col_indices_h.data(),
        num_src_vertices, num_dst_vertices, num_edges,
        out_row_offsets.data(), out_col_indices.data());
    
    EXPECT_EQ(out_row_offsets, expected_row_offsets);
    EXPECT_EQ(out_col_indices, expected_col_indices);
}

TEST_F(GraphTransposeTest, GPU_Transpose_WithCOO) {
    cudaError_t err = graph_transpose::transpose_csr_gpu(
        d_row_offsets, d_col_indices,
        num_src_vertices, num_dst_vertices, num_edges,
        d_out_row_offsets, d_out_col_indices,
        true, 0);
    
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
    
    std::vector<int> out_row_offsets(num_dst_vertices + 1);
    std::vector<int> out_col_indices(num_edges);
    
    cudaMemcpy(out_row_offsets.data(), d_out_row_offsets,
               (num_dst_vertices + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_col_indices.data(), d_out_col_indices,
               num_edges * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    EXPECT_EQ(out_row_offsets, expected_row_offsets);
    EXPECT_EQ(out_col_indices, expected_col_indices);
}

TEST_F(GraphTransposeTest, GPU_Transpose_WithoutCOO) {
    cudaError_t err = graph_transpose::transpose_csr_gpu(
        d_row_offsets, d_col_indices,
        num_src_vertices, num_dst_vertices, num_edges,
        d_out_row_offsets, d_out_col_indices,
        false, 0);
    
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
    
    std::vector<int> out_row_offsets(num_dst_vertices + 1);
    std::vector<int> out_col_indices(num_edges);
    
    cudaMemcpy(out_row_offsets.data(), d_out_row_offsets,
               (num_dst_vertices + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_col_indices.data(), d_out_col_indices,
               num_edges * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    EXPECT_EQ(out_row_offsets, expected_row_offsets);
    EXPECT_EQ(out_col_indices, expected_col_indices);
}

TEST_F(GraphTransposeTest, GPU_GlobalSort_Transpose) {
    cudaError_t err = graph_transpose::transpose_csr_gpu_global_sort(
        d_row_offsets, d_col_indices,
        num_src_vertices, num_dst_vertices, num_edges,
        d_out_row_offsets, d_out_col_indices,
        0);
    
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
    
    std::vector<int> out_row_offsets(num_dst_vertices + 1);
    std::vector<int> out_col_indices(num_edges);
    
    cudaMemcpy(out_row_offsets.data(), d_out_row_offsets,
               (num_dst_vertices + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_col_indices.data(), d_out_col_indices,
               num_edges * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    EXPECT_EQ(out_row_offsets, expected_row_offsets);
    EXPECT_EQ(out_col_indices, expected_col_indices);
}

TEST_F(GraphTransposeTest, GPU_KVSort_Transpose) {
    cudaError_t err = graph_transpose::transpose_csr_gpu_kv_sort(
        d_row_offsets, d_col_indices,
        num_src_vertices, num_dst_vertices, num_edges,
        d_out_row_offsets, d_out_col_indices,
        0);
    
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
    
    std::vector<int> out_row_offsets(num_dst_vertices + 1);
    std::vector<int> out_col_indices(num_edges);
    
    cudaMemcpy(out_row_offsets.data(), d_out_row_offsets,
               (num_dst_vertices + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_col_indices.data(), d_out_col_indices,
               num_edges * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    EXPECT_EQ(out_row_offsets, expected_row_offsets);
    EXPECT_EQ(out_col_indices, expected_col_indices);
}

TEST_F(GraphTransposeTest, AllMethodsAgree) {
    // CPU reference
    std::vector<int> cpu_row_offsets(num_dst_vertices + 1);
    std::vector<int> cpu_col_indices(num_edges);
    graph_transpose::transpose_csr_cpu(
        row_offsets_h.data(), col_indices_h.data(),
        num_src_vertices, num_dst_vertices, num_edges,
        cpu_row_offsets.data(), cpu_col_indices.data());
    
    // GPU with COO
    std::vector<int> gpu_coo_row_offsets(num_dst_vertices + 1);
    std::vector<int> gpu_coo_col_indices(num_edges);
    graph_transpose::transpose_csr_gpu(
        d_row_offsets, d_col_indices,
        num_src_vertices, num_dst_vertices, num_edges,
        d_out_row_offsets, d_out_col_indices,
        true, 0);
    cudaMemcpy(gpu_coo_row_offsets.data(), d_out_row_offsets,
               (num_dst_vertices + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_coo_col_indices.data(), d_out_col_indices,
               num_edges * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    // GPU without COO
    std::vector<int> gpu_nocoo_row_offsets(num_dst_vertices + 1);
    std::vector<int> gpu_nocoo_col_indices(num_edges);
    graph_transpose::transpose_csr_gpu(
        d_row_offsets, d_col_indices,
        num_src_vertices, num_dst_vertices, num_edges,
        d_out_row_offsets, d_out_col_indices,
        false, 0);
    cudaMemcpy(gpu_nocoo_row_offsets.data(), d_out_row_offsets,
               (num_dst_vertices + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_nocoo_col_indices.data(), d_out_col_indices,
               num_edges * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    // GPU global sort
    std::vector<int> gpu_global_row_offsets(num_dst_vertices + 1);
    std::vector<int> gpu_global_col_indices(num_edges);
    graph_transpose::transpose_csr_gpu_global_sort(
        d_row_offsets, d_col_indices,
        num_src_vertices, num_dst_vertices, num_edges,
        d_out_row_offsets, d_out_col_indices,
        0);
    cudaMemcpy(gpu_global_row_offsets.data(), d_out_row_offsets,
               (num_dst_vertices + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_global_col_indices.data(), d_out_col_indices,
               num_edges * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    // GPU kv sort
    std::vector<int> gpu_kv_row_offsets(num_dst_vertices + 1);
    std::vector<int> gpu_kv_col_indices(num_edges);
    graph_transpose::transpose_csr_gpu_kv_sort(
        d_row_offsets, d_col_indices,
        num_src_vertices, num_dst_vertices, num_edges,
        d_out_row_offsets, d_out_col_indices,
        0);
    cudaMemcpy(gpu_kv_row_offsets.data(), d_out_row_offsets,
               (num_dst_vertices + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_kv_col_indices.data(), d_out_col_indices,
               num_edges * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    // All methods should produce identical results
    EXPECT_TRUE(compare_csr(cpu_row_offsets, cpu_col_indices,
                           gpu_coo_row_offsets, gpu_coo_col_indices))
        << "GPU with COO differs from CPU";
    EXPECT_TRUE(compare_csr(cpu_row_offsets, cpu_col_indices,
                           gpu_nocoo_row_offsets, gpu_nocoo_col_indices))
        << "GPU without COO differs from CPU";
    EXPECT_TRUE(compare_csr(cpu_row_offsets, cpu_col_indices,
                           gpu_global_row_offsets, gpu_global_col_indices))
        << "GPU global sort differs from CPU";
    EXPECT_TRUE(compare_csr(cpu_row_offsets, cpu_col_indices,
                           gpu_kv_row_offsets, gpu_kv_col_indices))
        << "GPU kv sort differs from CPU";
}

TEST(GraphTransposeEdgeCases, EmptyGraph) {
    std::vector<int> row_offsets = {0, 0, 0};
    std::vector<int> col_indices;
    std::vector<int> out_row_offsets(3);
    std::vector<int> out_col_indices;
    
    graph_transpose::transpose_csr_cpu(
        row_offsets.data(), col_indices.data(),
        2, 2, 0,
        out_row_offsets.data(), out_col_indices.data());
    
    std::vector<int> expected_offsets = {0, 0, 0};
    EXPECT_EQ(out_row_offsets, expected_offsets);
}

TEST(GraphTransposeEdgeCases, SingleEdge) {
    std::vector<int> row_offsets = {0, 1, 1};
    std::vector<int> col_indices = {1};
    std::vector<int> out_row_offsets(3);
    std::vector<int> out_col_indices(1);
    
    graph_transpose::transpose_csr_cpu(
        row_offsets.data(), col_indices.data(),
        2, 2, 1,
        out_row_offsets.data(), out_col_indices.data());
    
    std::vector<int> expected_offsets = {0, 0, 1};
    std::vector<int> expected_cols = {0};
    EXPECT_EQ(out_row_offsets, expected_offsets);
    EXPECT_EQ(out_col_indices, expected_cols);
}

TEST(GraphTransposeEdgeCases, SelfLoops) {
    std::vector<int> row_offsets = {0, 1, 2, 3};
    std::vector<int> col_indices = {0, 1, 2};
    std::vector<int> out_row_offsets(4);
    std::vector<int> out_col_indices(3);
    
    graph_transpose::transpose_csr_cpu(
        row_offsets.data(), col_indices.data(),
        3, 3, 3,
        out_row_offsets.data(), out_col_indices.data());
    
    std::vector<int> expected_offsets = {0, 1, 2, 3};
    std::vector<int> expected_cols = {0, 1, 2};
    EXPECT_EQ(out_row_offsets, expected_offsets);
    EXPECT_EQ(out_col_indices, expected_cols);
}

TEST(GraphTransposeEdgeCases, DenseColumn) {
    // All vertices point to vertex 0
    std::vector<int> row_offsets = {0, 1, 2, 3, 4};
    std::vector<int> col_indices = {0, 0, 0, 0};
    std::vector<int> out_row_offsets(2);
    std::vector<int> out_col_indices(4);
    
    graph_transpose::transpose_csr_cpu(
        row_offsets.data(), col_indices.data(),
        4, 1, 4,
        out_row_offsets.data(), out_col_indices.data());
    
    std::vector<int> expected_offsets = {0, 4};
    std::vector<int> expected_cols = {0, 1, 2, 3};
    EXPECT_EQ(out_row_offsets, expected_offsets);
    EXPECT_EQ(out_col_indices, expected_cols);
}
