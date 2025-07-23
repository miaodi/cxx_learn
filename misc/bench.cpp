#include "Vector.h"
#include "func.h"
#include <benchmark/benchmark.h>
#include <functional>
#include <map>
#include <mutex>
#include <vector>

static void BM_TwiddleOrigin(benchmark::State &state) {
  std::uint64_t u = 123456789, v = 987654321;
  std::uint64_t result = 0;
  for (auto _ : state) {
    for (size_t i = 0; i < state.range(0); ++i) {
      result ^= MCRand::twiddle_origin(u, v);
      ++u;
      ++v;
      benchmark::DoNotOptimize(result);
    }
  }
}

static void BM_TwiddleNew(benchmark::State &state) {
  std::uint64_t u = 123456789, v = 987654321;
  std::uint64_t result = 0;
  for (auto _ : state) {
    for (size_t i = 0; i < state.range(0); ++i) {
      result ^= MCRand::twiddle_new(u, v);
      ++u;
      ++v;
      benchmark::DoNotOptimize(result);
    }
  }
}

BENCHMARK(BM_TwiddleOrigin)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_TwiddleNew)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

static void BM_tmcRand(benchmark::State &state) {
  std::uint32_t seed = 123456789;
  MCRand::tmcRand gen(seed);
  for (auto _ : state) {
    for (size_t i = 0; i < state.range(0); ++i) {
      benchmark::DoNotOptimize(gen.drand());
    }
  }
}

static void BM_tmcRandAVX2(benchmark::State &state) {
  std::uint32_t seed = 123456789;
  MCRand::tmcRandAVX2 gen(seed);
  for (auto _ : state) {
    for (size_t i = 0; i < state.range(0); ++i) {
      benchmark::DoNotOptimize(gen.drand());
    }
  }
}

BENCHMARK(BM_tmcRand)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_tmcRandAVX2)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

// test dynamic cast vs static cast

class Base {
public:
  Base(const int val = 0) : _value{val} {}
  virtual ~Base() = default; // Ensure a virtual destructor for dynamic_cast

  int _value;
};

class Derived1 : public Base {
public:
  Derived1(const int val = 1) : Base(val) {}
};

class Derived2 : public Base {
public:
  Derived2(const int val = 2) : Base(val) {}
};

class Derived3 : public Base {
public:
  Derived3(const int val = 3) : Base(val) {}
};

static std::map<int,
                std::pair<std::vector<int>, std::vector<std::unique_ptr<Base>>>>
    cast_map;

std::pair<std::vector<int>, std::vector<std::unique_ptr<Base>>> &
getCast(const int size) {
  if (cast_map.find(size) == cast_map.end()) {
    std::vector<int> values(size);
    std::vector<std::unique_ptr<Base>> bases(size);
    for (int i = 0; i < size; ++i) {
      values[i] = rand() % 3; // Randomly select 0, 1, or 2
      switch (values[i]) {
      case 0:
        bases[i] = std::make_unique<Derived1>(i);
        break;
      case 1:
        bases[i] = std::make_unique<Derived2>(i);
        break;
      case 2:
        bases[i] = std::make_unique<Derived3>(i);
        break;
      }
    }
    cast_map[size] = {values, std::move(bases)};
  }
  return cast_map[size];
}

static std::vector<std::function<void(std::unique_ptr<Base> &)>>
    static_cast_functions = {
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(static_cast<Derived1 *>(base.get())->_value);
        },
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(static_cast<Derived2 *>(base.get())->_value);
        },
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(static_cast<Derived3 *>(base.get())->_value);
        }};

static std::vector<std::function<void(std::unique_ptr<Base> &)>>
    dynamic_cast_functions = {
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(
              dynamic_cast<Derived1 *>(base.get())->_value);
        },
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(
              dynamic_cast<Derived2 *>(base.get())->_value);
        },
        [](std::unique_ptr<Base> &base) {
          benchmark::DoNotOptimize(
              dynamic_cast<Derived3 *>(base.get())->_value);
        }};

static void BM_CastStatic(benchmark::State &state) {
  auto &cast_data = getCast(state.range(0));
  auto &values = cast_data.first;
  auto &bases = cast_data.second;

  for (auto _ : state) {
    for (size_t i = 0; i < bases.size(); ++i) {
      static_cast_functions[values[i]](bases[i]);
    }
  }
}

static void BM_CastDynamic(benchmark::State &state) {
  auto &cast_data = getCast(state.range(0));
  auto &values = cast_data.first;
  auto &bases = cast_data.second;

  for (auto _ : state) {
    for (size_t i = 0; i < bases.size(); ++i) {
      dynamic_cast_functions[values[i]](bases[i]);
    }
  }
}

BENCHMARK(BM_CastStatic)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_CastDynamic)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

const int vec_size = 100;
std::vector<misc::MCVector> vec_a(vec_size);
std::vector<misc::MCVectorAVX2> vec_a_avx(vec_size);
std::vector<misc::MCVector> vec_b(vec_size);
std::vector<misc::MCVectorAVX2> vec_b_avx(vec_size);
std::once_flag vec_init_flag;
void initialize_vectors() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(-1.0, 1.0);
  std::generate(vec_a.begin(), vec_a.end(),
                [&]() { return misc::MCVector(dis(gen), dis(gen), dis(gen)); });
  for (size_t i = 0; i < vec_size; ++i) {
    vec_a_avx[i] = vec_a[i];
  }
  std::generate(vec_b.begin(), vec_b.end(),
                [&]() { return misc::MCVector(dis(gen), dis(gen), dis(gen)); });
  for (size_t i = 0; i < vec_size; ++i) {
    vec_b_avx[i] = vec_b[i];
  }
}

static void BM_ConstructVec(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      misc::MCVector v(vec_a[i % vec_size]);
      benchmark::DoNotOptimize(v[0]);
    }
  }
}

static void BM_ConstructVecAVX2(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      misc::MCVectorAVX2 v(vec_a_avx[i % vec_size]);
      benchmark::DoNotOptimize(v[0]);
    }
  }
}

BENCHMARK(BM_ConstructVec)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_ConstructVecAVX2)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

static void BM_Neg(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  misc::MCVector neg;
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      neg = -vec_a[i % vec_size];
      benchmark::DoNotOptimize(neg[0]);
    }
  }
}

static void BM_NegAVX2(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  misc::MCVectorAVX2 neg;
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      neg = -vec_a_avx[i % vec_size];
      benchmark::DoNotOptimize(neg[0]);
    }
  }
}

static void BM_NegAVX2Candidate(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  misc::MCVectorAVX2 neg;
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      neg = vec_a_avx[i % vec_size].neg();
      benchmark::DoNotOptimize(neg[0]);
    }
  }
}

BENCHMARK(BM_Neg)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_NegAVX2)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_NegAVX2Candidate)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

static void BM_Length(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      benchmark::DoNotOptimize(vec_a[i % vec_size].length());
    }
  }
}

static void BM_LengthAVX2(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      benchmark::DoNotOptimize(vec_a_avx[i % vec_size].length());
    }
  }
}

BENCHMARK(BM_Length)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_LengthAVX2)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

static void BM_DotProd(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      benchmark::DoNotOptimize(
          misc::dotProduct(vec_a[i % vec_size], vec_b[i % vec_size]));
    }
  }
}

static void BM_DotProdAVX2(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      benchmark::DoNotOptimize(
          misc::dotProduct(vec_a_avx[i % vec_size], vec_b_avx[i % vec_size]));
    }
  }
}

BENCHMARK(BM_DotProd)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_DotProdAVX2)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

static void BM_CrossProd(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  misc::MCVector cross_prod;
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      cross_prod = vec_a[i % vec_size] ^ vec_b[i % vec_size];
      benchmark::DoNotOptimize(cross_prod.ve);
    }
  }
}

static void BM_CrossProdAVX2(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  misc::MCVectorAVX2 cross_prod;
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      cross_prod = vec_a_avx[i % vec_size] ^ vec_b_avx[i % vec_size];
      benchmark::DoNotOptimize(cross_prod._ve.ve);
    }
  }
}

BENCHMARK(BM_CrossProd)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_CrossProdAVX2)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

static void BM_CrossProdLen(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      benchmark::DoNotOptimize(
          misc::crossProductLen(vec_a[i % vec_size], vec_b[i % vec_size]));
    }
  }
}

static void BM_CrossProdLenAVX2(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      benchmark::DoNotOptimize(
          vec_a_avx[i % vec_size].crossProductLen(vec_b_avx[i % vec_size]));
    }
  }
}

BENCHMARK(BM_CrossProdLen)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_CrossProdLenAVX2)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

static void BM_Sum(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      misc::MCVector sum = vec_a[i % vec_size] + vec_b[i % vec_size];

      benchmark::DoNotOptimize(sum[0]);
    }
  }
}

static void BM_SumAVX2(benchmark::State &state) {
  std::call_once(vec_init_flag, initialize_vectors);
  size_t size = state.range(0);
  for (auto _ : state) {
    for (size_t i = 0; i < size; ++i) {
      auto sum = vec_a_avx[i % vec_size] + vec_b_avx[i % vec_size];
      benchmark::DoNotOptimize(sum[0]);
    }
  }
}

BENCHMARK(BM_Sum)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);
BENCHMARK(BM_SumAVX2)->RangeMultiplier(16)->Range(1 << 0, 1 << 20);

BENCHMARK_MAIN();