#include "merge.cuh"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>
#include <random>
#include <vector>

void make_sorted_input(std::vector<int> &out, int size, int min_val, int max_val,
                       std::mt19937 &rng) {
  std::uniform_int_distribution<int> dist(min_val, max_val);
  out.resize(size);
  for (int i = 0; i < size; ++i) {
    out[i] = dist(rng);
  }
  std::sort(out.begin(), out.end());
}

void set_metrics(benchmark::State &state, int size_a, int size_b) {
  int total = size_a + size_b;
  int64_t bytes = static_cast<int64_t>(total) * sizeof(int) * 3;
  state.SetBytesProcessed(bytes * static_cast<int64_t>(state.iterations()));
  state.SetItemsProcessed(static_cast<int64_t>(total) * state.iterations());
}

// Cache one (size_a, size_b) worth of data so CPU/GPU/GPU_Shared for the same
// size reuse one vector pair (a, b) and device buffers. Cleared when size changes.
struct CachedMergeData {
  std::vector<int> a, b, out;
  int *d_a = nullptr;
  int *d_b = nullptr;
  int *d_out = nullptr;
  void free_device() {
    if (d_a) {
      cudaFree(d_a);
      d_a = nullptr;
    }
    if (d_b) {
      cudaFree(d_b);
      d_b = nullptr;
    }
    if (d_out) {
      cudaFree(d_out);
      d_out = nullptr;
    }
  }
  ~CachedMergeData() { free_device(); }
};

static CachedMergeData s_cache;
static int s_cached_size_a = -1;
static int s_cached_size_b = -1;

struct CacheCleanup {
  ~CacheCleanup() {
    s_cache.free_device();
    s_cache.a.clear();
    s_cache.b.clear();
    s_cache.out.clear();
    s_cached_size_a = -1;
    s_cached_size_b = -1;
  }
} s_cache_cleanup;

class MergeBench : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    int size_a = static_cast<int>(state.range(0));
    int size_b = static_cast<int>(state.range(1));

    if (size_a == s_cached_size_a && size_b == s_cached_size_b) {
      // Reuse cached data for this size (same size, run CPU then GPU then GPU_Shared).
      a.swap(s_cache.a);
      b.swap(s_cache.b);
      out.swap(s_cache.out);
      d_a = s_cache.d_a;
      d_b = s_cache.d_b;
      d_out = s_cache.d_out;
      s_cache.d_a = s_cache.d_b = s_cache.d_out = nullptr;
      return;
    }

    // New size: free previous cache and create fresh data.
    if (s_cached_size_a >= 0) {
      s_cache.free_device();
      s_cache.a.clear();
      s_cache.b.clear();
      s_cache.out.clear();
    }
    s_cached_size_a = size_a;
    s_cached_size_b = size_b;

    std::mt19937 rng(2028);
    make_sorted_input(a, size_a, -100000, 100000, rng);
    make_sorted_input(b, size_b, -100000, 100000, rng);
    out.resize(size_a + size_b);

    int total = size_a + size_b;
    cudaMalloc(&d_a, size_a * sizeof(int));
    cudaMalloc(&d_b, size_b * sizeof(int));
    cudaMalloc(&d_out, total * sizeof(int));
    if (size_a > 0) {
      cudaMemcpy(d_a, a.data(), size_a * sizeof(int), cudaMemcpyHostToDevice);
    }
    if (size_b > 0) {
      cudaMemcpy(d_b, b.data(), size_b * sizeof(int), cudaMemcpyHostToDevice);
    }
  }

  void TearDown(const ::benchmark::State &) override {
    // Return data to cache so next run (GPU/GPU_Shared same size) can reuse.
    s_cache.a.swap(a);
    s_cache.b.swap(b);
    s_cache.out.swap(out);
    s_cache.d_a = d_a;
    s_cache.d_b = d_b;
    s_cache.d_out = d_out;
    d_a = d_b = d_out = nullptr;
  }

protected:
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> out;
  int *d_a = nullptr;
  int *d_b = nullptr;
  int *d_out = nullptr;
};

// impl: 0 = CPU, 1 = GPU, 2 = GPU_Shared. Run order: one size then CPU/GPU/GPU_Shared
// so one vector pair (a,b) is created once per size and reused.
BENCHMARK_DEFINE_F(MergeBench, Even)(benchmark::State &state) {
  int size_a = static_cast<int>(state.range(0));
  int size_b = static_cast<int>(state.range(1));
  int impl = static_cast<int>(state.range(2));
  if (impl == 0) {
    for (auto _ : state) {
      std::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin());
      benchmark::DoNotOptimize(out.data());
    }
  } else if (impl == 1) {
    for (auto _ : state) {
      PMPP::merge_device(d_a, size_a, d_b, size_b, d_out);
      benchmark::DoNotOptimize(d_out);
    }
  } else if(impl == 2) {
    for (auto _ : state) {
      PMPP::merge_shared_device(d_a, size_a, d_b, size_b, d_out);
      benchmark::DoNotOptimize(d_out);
    }
  } else if(impl == 3) {
    for (auto _ : state) {
      PMPP::merge_shared_partitioned_device(d_a, size_a, d_b, size_b, d_out);
      benchmark::DoNotOptimize(d_out);
    }
  }
  set_metrics(state, size_a, size_b);
}

BENCHMARK_DEFINE_F(MergeBench, Uneven)(benchmark::State &state) {
  int size_a = static_cast<int>(state.range(0));
  int size_b = static_cast<int>(state.range(1));
  int impl = static_cast<int>(state.range(2));
  if (impl == 0) {
    for (auto _ : state) {
      std::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin());
      benchmark::DoNotOptimize(out.data());
    }
  } else if (impl == 1) {
    for (auto _ : state) {
      PMPP::merge_device(d_a, size_a, d_b, size_b, d_out);
      benchmark::DoNotOptimize(d_out);
    }
  } else if(impl == 2) {
    for (auto _ : state) {
      PMPP::merge_shared_device(d_a, size_a, d_b, size_b, d_out);
      benchmark::DoNotOptimize(d_out);
    }
  } else if(impl == 3) {
    for (auto _ : state) {
      PMPP::merge_shared_partitioned_device(d_a, size_a, d_b, size_b, d_out);
      benchmark::DoNotOptimize(d_out);
    }
  }
  set_metrics(state, size_a, size_b);
}

// One size at a time: for each size run CPU (0), GPU (1), GPU_Shared (2).
static void ApplyEvenSizesBySize(benchmark::internal::Benchmark *b) {
  for (int size : {1 << 12, 1 << 16, 1 << 20, 1 << 24, 1 << 28}) {
    b->Args({size, size, 0});
    b->Args({size, size, 1});
    b->Args({size, size, 2});
    b->Args({size, size, 3});
  }
}

static void ApplyUnevenSizesBySize(benchmark::internal::Benchmark *b) {
  for (int size : {1 << 12, 1 << 16, 1 << 20, 1 << 24, 1 << 28}) {
    int size_b = size / 2;
    b->Args({size, size_b, 0});
    b->Args({size, size_b, 1});
    b->Args({size, size_b, 2});
    b->Args({size, size_b, 3});
  }
}

void register_merge_benchmarks() {
  BENCHMARK_REGISTER_F(MergeBench, Even)
      ->Apply(ApplyEvenSizesBySize)
      ->ArgNames({"size_a", "size_b", "impl"})
      ->Unit(benchmark::kMicrosecond)
      ->UseRealTime();

  BENCHMARK_REGISTER_F(MergeBench, Uneven)
      ->Apply(ApplyUnevenSizesBySize)
      ->ArgNames({"size_a", "size_b", "impl"})
      ->Unit(benchmark::kMicrosecond)
      ->UseRealTime();
}

int main(int argc, char **argv) {
  // Explicitly register benchmarks so the linker can't dead-strip the
  // registration code when building with -ffunction-sections/--gc-sections.
  register_merge_benchmarks();

  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  return 0;
}
