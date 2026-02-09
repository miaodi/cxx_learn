#include "merge.cuh"

#include <algorithm>
#include <benchmark/benchmark.h>
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

class MergeBench : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State &state) override {
    int size_a = static_cast<int>(state.range(0));
    int size_b = static_cast<int>(state.range(1));
    std::mt19937 rng(2028);

    make_sorted_input(a, size_a, -100000, 100000, rng);
    make_sorted_input(b, size_b, -100000, 100000, rng);
    out.resize(size_a + size_b);
  }

  void TearDown(const ::benchmark::State &) override {
    a.clear();
    b.clear();
    out.clear();
  }

protected:
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> out;
};

BENCHMARK_DEFINE_F(MergeBench, CPU_Even)(benchmark::State &state) {
  int size_a = static_cast<int>(state.range(0));
  int size_b = static_cast<int>(state.range(1));
  for (auto _ : state) {
    std::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin());
    benchmark::DoNotOptimize(out.data());
  }
  set_metrics(state, size_a, size_b);
}

BENCHMARK_DEFINE_F(MergeBench, GPU_Even)(benchmark::State &state) {
  int size_a = static_cast<int>(state.range(0));
  int size_b = static_cast<int>(state.range(1));
  for (auto _ : state) {
    PMPP::merge(a.data(), size_a, b.data(), size_b, out.data());
    benchmark::DoNotOptimize(out.data());
  }
  set_metrics(state, size_a, size_b);
}

BENCHMARK_DEFINE_F(MergeBench, CPU_Uneven)(benchmark::State &state) {
  int size_a = static_cast<int>(state.range(0));
  int size_b = static_cast<int>(state.range(1));
  for (auto _ : state) {
    std::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin());
    benchmark::DoNotOptimize(out.data());
  }
  set_metrics(state, size_a, size_b);
}

BENCHMARK_DEFINE_F(MergeBench, GPU_Uneven)(benchmark::State &state) {
  int size_a = static_cast<int>(state.range(0));
  int size_b = static_cast<int>(state.range(1));
  for (auto _ : state) {
    PMPP::merge(a.data(), size_a, b.data(), size_b, out.data());
    benchmark::DoNotOptimize(out.data());
  }
  set_metrics(state, size_a, size_b);
}

BENCHMARK_DEFINE_F(MergeBench, GPU_Shared_Even)(benchmark::State &state) {
  int size_a = static_cast<int>(state.range(0));
  int size_b = static_cast<int>(state.range(1));
  for (auto _ : state) {
    PMPP::merge_shared(a.data(), size_a, b.data(), size_b, out.data());
    benchmark::DoNotOptimize(out.data());
  }
  set_metrics(state, size_a, size_b);
}

BENCHMARK_DEFINE_F(MergeBench, GPU_Shared_Uneven)(benchmark::State &state) {
  int size_a = static_cast<int>(state.range(0));
  int size_b = static_cast<int>(state.range(1));
  for (auto _ : state) {
    PMPP::merge_shared(a.data(), size_a, b.data(), size_b, out.data());
    benchmark::DoNotOptimize(out.data());
  }
  set_metrics(state, size_a, size_b);
}

// Helper applicators to attach argument sets in one registration call each.
static void ApplyEvenSizes(benchmark::internal::Benchmark *b) {
  for (int size :
       {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24}) {
    b->Args({size, size});
  }
}

static void ApplyUnevenSizes(benchmark::internal::Benchmark *b) {
  for (int size :
       {1 << 10, 1 << 12, 1 << 14, 1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24}) {
    b->Args({size, size / 2});
  }
}

void register_merge_benchmarks() {
  BENCHMARK_REGISTER_F(MergeBench, CPU_Even)
      ->Apply(ApplyEvenSizes)
      ->Unit(benchmark::kMicrosecond)
      ->UseRealTime();

  BENCHMARK_REGISTER_F(MergeBench, GPU_Even)
      ->Apply(ApplyEvenSizes)
      ->Unit(benchmark::kMicrosecond)
      ->UseRealTime();

  BENCHMARK_REGISTER_F(MergeBench, GPU_Shared_Even)
      ->Apply(ApplyEvenSizes)
      ->Unit(benchmark::kMicrosecond)
      ->UseRealTime();

  BENCHMARK_REGISTER_F(MergeBench, CPU_Uneven)
      ->Apply(ApplyUnevenSizes)
      ->Unit(benchmark::kMicrosecond)
      ->UseRealTime();

  BENCHMARK_REGISTER_F(MergeBench, GPU_Uneven)
      ->Apply(ApplyUnevenSizes)
      ->Unit(benchmark::kMicrosecond)
      ->UseRealTime();

  BENCHMARK_REGISTER_F(MergeBench, GPU_Shared_Uneven)
      ->Apply(ApplyUnevenSizes)
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
