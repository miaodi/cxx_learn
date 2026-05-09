# CUDA GEMM Learning Workspace

This folder is for rebuilding GEMM on CUDA from a simple baseline toward more
efficient kernels.

The implementations tracked here are:

```text
sgemm_ijk
sgemm_tiled_16
sgemm_tiled_16_2x2
sgemm_tiled_16_4x4
sgemm_tiled_16_8x8
sgemm_tiled_16_16x16
sgemm_tiled_16_2x2_coalesced
sgemm_tiled_16_2x2_k32_coalesced
```

The name follows the BLAS precision convention:

```text
s: single precision float
gemm: general matrix multiply
ijk: implementation loop/order label for this learning version
```

## Contract

Use the BLAS-style operation:

```text
C = alpha * A * B + beta * C
```

The current scope is row-major, no-transpose SGEMM:

```text
A: M x K
B: K x N
C: M x N
```

Elements are addressed as:

```cpp
A[row * lda + kk]
B[kk * ldb + col]
C[row * ldc + col]
```

For contiguous row-major matrices:

```text
lda = K
ldb = N
ldc = N
```

The API expects device pointers. Host allocation and host-device copies should
stay outside the kernel wrapper so tests, demos, and benchmarks can decide what
they want to measure.

## Implementations

`sgemm_ijk` assigns one CUDA thread to one output element:

```cpp
for (int kk = 0; kk < K; ++kk) {
  sum += A[row * lda + kk] * B[kk * ldb + col];
}
C[row * ldc + col] = alpha * sum + beta * C[row * ldc + col];
```

This is intentionally naive:

- Each thread reads one row of `A` contiguously.
- Each thread reads one column of `B` with stride `ldb`.
- Neighboring threads reuse the same `A` values but reload them independently.
- There is no shared-memory tiling yet.

That makes it a good correctness baseline and a poor performance target.

`sgemm_tiled_16` keeps the one-thread-per-output mapping, but stages one
`16x16` tile of `A` and one `16x16` tile of `B` through shared memory for each
chunk of the `K` dimension.

For each block:

```text
threads/block: 16 x 16 = 256
C outputs/block: 16 x 16 = 256
shared memory: (16 x 16 + 16 x 16) floats = 2 KiB
```

`sgemm_tiled_16_2x2` keeps the same 256 threads per block, but each thread
computes a `2x2` patch of `C` in registers:

```text
thread output:

C[row + 0, col + 0]  C[row + 0, col + 1]
C[row + 1, col + 0]  C[row + 1, col + 1]
```

That gives each thread four accumulators:

```cpp
float c00 = 0.0f;
float c01 = 0.0f;
float c10 = 0.0f;
float c11 = 0.0f;
```

For each block:

```text
threads/block: 16 x 16 = 256
C outputs/block: 32 x 32 = 1024
shared A tile: 32 x 16 floats
shared B tile: 16 x 32 floats
shared memory: (32 x 16 + 16 x 32) floats = 4 KiB
```

The reason this is the next step after simple tiling is reuse: each loaded
`A` value contributes to two columns of `C`, and each loaded `B` value
contributes to two rows of `C`.

`sgemm_tiled_16_4x4` increases the register blocking again. It keeps the same
`16x16` CUDA thread block, but each thread computes a `4x4` patch of `C`:

```text
threads/block: 16 x 16 = 256
C outputs/block: 64 x 64 = 4096
shared A tile: 64 x 16 floats
shared B tile: 16 x 64 floats
shared memory: (64 x 16 + 16 x 64) floats = 8 KiB
accumulators/thread: 16
```

This does more arithmetic per loaded shared-memory tile than `2x2`, but it also
uses more registers per thread. Whether it wins depends on register pressure,
occupancy, and how well the compiler keeps the accumulator array in registers.

`sgemm_tiled_16_8x8` and `sgemm_tiled_16_16x16` keep increasing work per
thread to find where performance decays:

```text
variant    C tile     shared A    shared B    shared memory    accumulators/thread
8x8        128 x 128  128 x 16    16 x 128    16 KiB           64
16x16      256 x 256  256 x 16    16 x 256    32 KiB           256
```

These variants intentionally push register pressure and reduce the number of
thread blocks for a fixed matrix. `16x16` is especially likely to spill
accumulators to local memory or lose occupancy, but it is useful as a decay
point in the benchmark curve.

`sgemm_tiled_16_2x2_coalesced` keeps the same output tile and same register
blocking, but changes the shared-memory load pattern. Instead of each compute
thread loading its own two `A` values and two `B` values, the block uses flat
cooperative indexing over the full shared tiles:

```text
A shared tile: 32 x 16 = 512 floats
B shared tile: 16 x 32 = 512 floats
threads/block: 256
loads/thread/tile: 2 A values + 2 B values
```

This makes `B` loads fully contiguous at the warp level because each warp maps
to one `32`-float row of the `B` tile. `A` loads are still split into two
contiguous half-warp segments because the `A` tile row is only `16` floats wide.
That is an improvement over the direct `2x2` loader while keeping the same
compute kernel shape.

`sgemm_tiled_16_2x2_k32_coalesced` keeps the same `32x32` output tile, but
doubles the K tile from `16` to `32`:

```text
A shared tile: 32 x 32 = 1024 floats
B shared tile: 32 x 32 = 1024 floats
threads/block: 256
loads/thread/tile: 4 A values + 4 B values
shared memory: 8 KiB
```

This makes `A` loading more effective for row-major matrices because a warp can
load one contiguous `32`-float row of the `A` K slice. It also halves the number
of K-tile loop iterations. The tradeoff is twice the shared-memory load work per
iteration and twice the shared-memory footprint compared with the K16 coalesced
variant.

## Planned Progression

1. `sgemm_ijk`: one thread per `C` element, direct global-memory reads.
2. `sgemm_tiled_16`: cooperative square shared-memory tiles of `A` and `B`.
3. `sgemm_tiled_16_2x2`: register blocking, four `C` elements per thread.
4. `sgemm_tiled_16_4x4`: register blocking, sixteen `C` elements per thread.
5. `sgemm_tiled_16_8x8`: register blocking, sixty-four `C` elements per thread.
6. `sgemm_tiled_16_16x16`: register blocking, 256 `C` elements per thread.
7. `sgemm_tiled_16_2x2_coalesced`: flat cooperative global-memory loads.
8. `sgemm_tiled_16_2x2_k32_coalesced`: doubled K tile for full-warp A loads.
9. Vectorized global loads where layout allows it.
10. Compare against cuBLAS for a performance ceiling.

## Benchmark

`PMPP_gemm_bench` compares one fixed moderate case:

```text
M = 1024
N = 1024
K = 1024
```

It runs:

```text
SGEMM/NaiveIjk/1024
SGEMM/Tiled16/1024
SGEMM/Tiled16_2x2/1024
SGEMM/Tiled16_4x4/1024
SGEMM/Tiled16_8x8/1024
SGEMM/Tiled16_16x16/1024
SGEMM/Tiled16_2x2Coalesced/1024
SGEMM/Tiled16_2x2K32Coalesced/1024
SGEMM/cuBLAS/1024
```

The benchmark cases keep matrices on the device and synchronize after each
call, so the measurement excludes host allocation and host-device copies. The
cuBLAS call uses the row-major identity:

```text
C = A * B  <=>  C^T = B^T * A^T
```

Build and run the focused targets:

```bash
cmake --build release --target PMPP_gemm_test PMPP_gemm_bench
./release/cuda/PMPP/gemm/PMPP_gemm_test
./release/cuda/PMPP/gemm/PMPP_gemm_bench --benchmark_min_time=0.2s
```

## Example Result

The following sample is from a `1024 x 1024 x 1024` run before
`sgemm_tiled_16_2x2` was added. Hardware, clocks, CUDA version, and cuBLAS math
mode can change these numbers.

| Kernel | Time | Throughput |
| --- | ---: | ---: |
| `SGEMM/NaiveIjk/1024` | 9.55 ms | 224.981 GFLOP/s |
| `SGEMM/Tiled16/1024` | 4.21 ms | 509.975 GFLOP/s |
| `SGEMM/cuBLAS/1024` | 0.735 ms | 2.92338 TFLOP/s |

![SGEMM 1024 benchmark plot](sgemm_1024_benchmark.svg)

From that sample:

```text
Tiled16 speedup over NaiveIjk: 9.55 / 4.21 = 2.27x
cuBLAS speedup over Tiled16: 4.21 / 0.735 = 5.73x
cuBLAS speedup over NaiveIjk: 9.55 / 0.735 = 12.99x
```

After running the current benchmark, add the `SGEMM/Tiled16_2x2/1024`,
`SGEMM/Tiled16_4x4/1024`, `SGEMM/Tiled16_8x8/1024`,
`SGEMM/Tiled16_16x16/1024`, `SGEMM/Tiled16_2x2Coalesced/1024`, and
`SGEMM/Tiled16_2x2K32Coalesced/1024` rows to the table and regenerate the plot
with the new timing.

## CUTLASS Example

`cutlass_sgemm_example.cu` is a minimal row-major SGEMM example using
`cutlass::gemm::device::Gemm`. The source is intentionally commented around the
important CUTLASS concepts: layout selection, the GEMM type alias,
`Arguments`, `can_implement`, and the asynchronous launch call.

It computes:

```text
D = alpha * A * B + beta * C

A: 128 x 64 row-major
B: 64 x 96 row-major
C: 128 x 96 row-major
D: 128 x 96 row-major
```

The example mirrors the learning kernels in this folder by using row-major
inputs and explicit leading dimensions:

```cpp
using RowMajor = cutlass::layout::RowMajor;
using CutlassSgemm =
    cutlass::gemm::device::Gemm<float, RowMajor, float, RowMajor, float,
                                RowMajor, float>;

CutlassSgemm::Arguments args(
    {M, N, K},
    {d_A, K},
    {d_B, N},
    {d_C, N},
    {d_D, N},
    {alpha, beta});
```

CMake only builds the example when CUTLASS headers are available. You can point
`CUTLASS_ROOT` at an existing CUTLASS checkout:

```bash
cmake -S . -B release -DUSE_CUDA=ON -DCUTLASS_ROOT=/path/to/cutlass
cmake --build release --target PMPP_cutlass_sgemm_example
./release/cuda/PMPP/gemm/PMPP_cutlass_sgemm_example
```

Or let CMake download CUTLASS with `FetchContent`:

```bash
cmake -S . -B release -DUSE_CUDA=ON -DCXX_LEARNING_FETCH_CUTLASS=ON
cmake --build release --target PMPP_cutlass_sgemm_example
./release/cuda/PMPP/gemm/PMPP_cutlass_sgemm_example
```

The fetched version is controlled by:

```bash
-DCXX_LEARNING_CUTLASS_GIT_TAG=v4.3.4
```

Expected output shape:

```text
CUTLASS row-major SGEMM example
M=128 N=96 K=64
max_abs_error=<small number>
```
