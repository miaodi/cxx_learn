# Register Latency Benchmark

This benchmark asks whether splitting a random-access sum across multiple
independent accumulators can reduce loop-carried dependency pressure and expose
more independent work to the CPU out-of-order engine.

The baseline is conceptually:

```cpp
sum += a[perm[i]];
```

Every add depends on the previous value of `sum`, and every random data load
depends on first loading `perm[i]`. The multi-accumulator variants split the
reduction into independent chains such as:

```cpp
sum0 += a[perm[i + 0]];
sum1 += a[perm[i + 1]];
sum2 += a[perm[i + 2]];
sum3 += a[perm[i + 3]];
```

The partial sums are combined at the end.

## What This Measures

This experiment focuses on scalar latency hiding in a random-access reduction.

- A single accumulator creates a loop-carried dependency chain: each addition
  must wait for the previous accumulator value.
- Random indexing adds another latency source: each `values[perm[i]]` load waits
  for the corresponding permutation entry before the data address is known.
- Hardware prefetchers are usually weak on random permutations, so larger arrays
  often become latency-bound rather than bandwidth-bound.
- Multiple accumulators create independent dependency chains, giving
  out-of-order execution more loads and additions to overlap.
- More accumulators can expose memory-level parallelism, but only until the CPU
  runs into limits such as register pressure, instruction scheduling, cache
  misses, TLB misses, or the maximum number of outstanding memory operations.

## Benchmark Design

The benchmark is implemented in `bench.cpp` and built as
`register_latency_bench`.

- The value type is `std::uint64_t`.
- The tested value-array sizes are 4 KiB, 32 KiB, 256 KiB, 2 MiB, 16 MiB, and
  64 MiB.
- The tested accumulator counts are 1, 2, 4, 8, and 16.
- Each benchmark iteration sums the whole value array through a shuffled
  permutation array.
- The same deterministic random permutation is used for all accumulator counts
  at the same array size.
- The permutation seed depends on the array length, not on the accumulator
  count, so accumulator variants are compared against the same access pattern.
- Auto-vectorization is disabled for this target with compiler-specific flags to
  keep the experiment focused on scalar dependency chains instead of SIMD gather
  behavior.
- `items_per_second` reports processed `std::uint64_t` values per second.
- `bytes_per_second` reports processed value bytes plus permutation-index bytes
  per second.
- The custom `accumulators` counter records the accumulator count for that run.
- The `array=...` label reports the value-array size.

## How To Run

Configure the repository in release mode with benchmarks enabled through fetched
dependencies:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCXX_LEARNING_BUILD_TESTS=OFF -DCXX_LEARNING_FETCH_DEPS=ON
```

Build only this benchmark target:

```sh
cmake --build build --target register_latency_bench -j
```

Run the full benchmark:

```sh
./build/RegisterLatency/register_latency_bench --benchmark_min_time=0.5s
```

Run only one array size, for example 64 MiB:

```sh
./build/RegisterLatency/register_latency_bench --benchmark_min_time=0.5s --benchmark_filter='BM_RandomAccessSumAccumulators.*/array_bytes:67108864'
```

## Example Result

The following summary is based on one run of the benchmark. Times are the
reported wall-clock `Time` values, rounded for readability.

| Array size | 1 accumulator | Best result | Approx. change vs 1 accumulator |
| --- | ---: | ---: | ---: |
| 4 KiB | 124 ns | 8 accumulators, 119 ns | 4% faster |
| 32 KiB | 1248 ns | 8 accumulators, 1024 ns | 18% faster |
| 256 KiB | 13.59 us | 4 accumulators, 12.94 us | 5% faster |
| 2 MiB | 249.26 us | 4 accumulators, 242.82 us | 3% faster |
| 16 MiB | 5.89 ms | 8 accumulators, 5.80 ms | 2% faster |
| 64 MiB | 56.50 ms | 8 accumulators, 51.85 ms | 8% faster |

A compact view of throughput from the same run:

| Array size | Best accumulator count | Best items/s | Best bytes/s |
| --- | ---: | ---: | ---: |
| 4 KiB | 8 | 4.29 G/s | 63.97 GiB/s |
| 32 KiB | 8 | 4.00 G/s | 59.61 GiB/s |
| 256 KiB | 4 | 2.53 G/s | 37.73 GiB/s |
| 2 MiB | 4 | 1.08 G/s | 16.09 GiB/s |
| 16 MiB | 8 | 361.89 M/s | 5.39 GiB/s |
| 64 MiB | 8 | 161.80 M/s | 2.41 GiB/s |

## Interpretation

The result is useful precisely because it is mixed rather than monotonic.

- Small arrays fit in cache, so the benchmark is less dominated by long memory
  latency and more sensitive to loop overhead, instruction scheduling, and
  measurement noise.
- Moderate sizes can show benefits from breaking the single accumulator chain,
  but the extra instructions and registers can also offset those gains.
- Large random data can benefit from multiple independent misses in flight,
  because the CPU has more opportunities to overlap cache misses and arithmetic.
- The benefit saturates once the core reaches hardware limits such as load-buffer
  capacity, reorder-buffer capacity, TLB reach, cache miss handling resources, or
  memory-system latency.
- Sixteen accumulators are not automatically better; they can increase register
  pressure, code size, and scheduling complexity enough to lose to 4 or 8
  accumulators.

## Caveats

- Benchmark results can vary from run to run because of OS scheduling,
  interrupts, background work, and timer noise.
- CPU frequency scaling, turbo behavior, and thermal limits can change the
  measured time.
- Cache hierarchy, TLB size, memory latency, and memory-level parallelism vary
  significantly by CPU microarchitecture.
- Compiler version and optimization choices can change instruction scheduling,
  unrolling, register allocation, and code size.
- Integer reduction has exact wraparound behavior for `std::uint64_t`, while
  floating-point reductions have order-dependent rounding semantics; changing the
  value type can change both correctness considerations and performance.
