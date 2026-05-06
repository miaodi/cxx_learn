# GEMM Learning Workspace

This directory is a workspace for learning how GEMM implementations are built
and optimized.

GEMM means general matrix multiplication. The standard operation is:

```text
C = alpha * A * B + beta * C
```

The active kernels in `gemm.hpp` are:

```text
gemm_naive
gemm_ikj
gemm_blocked
gemm_packed_b
gemm_packed_ab
gemm_packed_ab_register_blocked
gemm_packed_ab_register_blocked_avx2_float
gemm_packed_ab_prepack_a
```

They all use this row-major BLAS-style contract.

## Current Scope

The current canonical API is:

```cpp
template <typename T>
void gemm_naive(int M, int N, int K,
                T alpha,
                const T *A, int lda,
                const T *B, int ldb,
                T beta,
                T *C, int ldc);

template <typename T>
void gemm_ikj(int M, int N, int K,
              T alpha,
              const T *A, int lda,
              const T *B, int ldb,
              T beta,
              T *C, int ldc);

template <typename T, int BM = 64, int BN = 64, int BK = 64>
void gemm_blocked(int M, int N, int K,
                  T alpha,
                  const T *A, int lda,
                  const T *B, int ldb,
                  T beta,
                  T *C, int ldc);

template <typename T, int BM = 64, int BN = 64, int BK = 64>
void gemm_packed_b(int M, int N, int K,
                   T alpha,
                   const T *A, int lda,
                   const T *B, int ldb,
                   T beta,
                   T *C, int ldc);

template <typename T, int BM = 64, int BN = 64, int BK = 64>
void gemm_packed_ab(int M, int N, int K,
                    T alpha,
                    const T *A, int lda,
                    const T *B, int ldb,
                    T beta,
                    T *C, int ldc);

template <typename T, int BM = 64, int BN = 64, int BK = 64>
void gemm_packed_ab_register_blocked(int M, int N, int K,
                                     T alpha,
                                     const T *A, int lda,
                                     const T *B, int ldb,
                                     T beta,
                                     T *C, int ldc);

template <int BM = 64, int BN = 64, int BK = 64>
void gemm_packed_ab_register_blocked_avx2_float(int M, int N, int K,
                                                float alpha,
                                                const float *A, int lda,
                                                const float *B, int ldb,
                                                float beta,
                                                float *C, int ldc);

template <typename T, int BM = 64, int BN = 64, int BK = 64>
void gemm_packed_ab_prepack_a(int M, int N, int K,
                              T alpha,
                              const T *A, int lda,
                              const T *B, int ldb,
                              T beta,
                              T *C, int ldc);
```

The AVX2 float variant requires `BN` to be a multiple of `8`, matching the
eight `float` lanes in one 256-bit AVX2 vector.

This computes a row-major, no-transpose GEMM:

```text
C = alpha * A * B + beta * C
```

with logical matrix shapes:

```text
A: M x K
B: K x N
C: M x N
```

## Row-Major Storage

Classic Fortran BLAS assumes column-major storage. This workspace currently uses
row-major storage because it is more natural for C and C++ arrays.

For this row-major interface, elements are addressed as:

```cpp
A[i * lda + k]
B[k * ldb + j]
C[i * ldc + j]
```

So for contiguous row-major matrices:

```text
lda = K
ldb = N
ldc = N
```

Example call for contiguous matrices:

```cpp
gemm_naive(M, N, K,
           T(1), A.data(), K,
           B.data(), N,
           T(0), C.data(), N);
```

## Implemented Kernels

### `gemm_naive`

Uses `i-j-k` loop order:

```cpp
for (int i = 0; i < M; ++i)
  for (int j = 0; j < N; ++j)
    for (int k = 0; k < K; ++k)
      sum += A[i * lda + k] * B[k * ldb + j];
```

This is a simple dot-product formulation. In row-major storage, the access to
`B[k * ldb + j]` moves down a column of `B` as `k` changes. That means each
access jumps by `ldb` elements, so the innermost loop has poor spatial locality
for `B`.

### `gemm_ikj`

Uses `i-k-j` loop order:

```cpp
for (int i = 0; i < M; ++i)
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      C[i * ldc + j] += alpha * A[i * lda + k] * B[k * ldb + j];
```

The main improvement is indeed cache behavior for `B`. With `j` in the
innermost loop, `B[k * ldb + j]` walks across one contiguous row of `B`, which
fits row-major storage. The same `A[i * lda + k]` scalar is also reused across
the whole inner `j` loop.

The tradeoff is that `gemm_ikj` updates `C` repeatedly as it accumulates over
`k`, so it first applies `beta` to `C` and then performs `+=` updates. This is
still useful because the updates walk contiguous rows of `C`, but it is not the
final form. Cache blocking is the next step that makes reuse of `B` and `C`
more controlled.

For the unblocked kernels, a simple worst-case memory model assumes each
multiply-add effectively brings in one value from `A` and one value from `B`:

```text
work         ~= 2 * M * N * K flops
A/B traffic  ~= 2 * M * N * K * sizeof(T) bytes
```

That gives a rough arithmetic intensity of:

```text
2 * M * N * K
----------------------------- = 1 / sizeof(T) flops/byte
2 * M * N * K * sizeof(T)
```

For `float`, this is:

```text
1 / 4 = 0.25 flops/byte
```

This is intentionally pessimistic because hardware caches and prefetchers still
reuse some data, especially in `gemm_ikj` where rows of `B` and `C` are accessed
contiguously. The point of the model is to show the direction: without blocking,
reuse is mostly accidental and cache-capacity dependent; with blocking, reuse is
made explicit.

### `gemm_blocked`

Adds cache blocking:

```cpp
for (int ii = 0; ii < M; ii += BM)
  for (int jj = 0; jj < N; jj += BN)
    for (int kk = 0; kk < K; kk += BK)
      update C[ii:ii+BM, jj:jj+BN]
      using  A[ii:ii+BM, kk:kk+BK]
      and    B[kk:kk+BK, jj:jj+BN]
```

The inner block computation still uses `i-k-j` order, but only inside one
smaller tile. The goal is to keep the active pieces of `A`, `B`, and `C` closer
to cache while accumulating a block of `C`.

The block sizes are separate because the dimensions have different jobs:

```text
BM: rows of the active A and C block
BN: columns of the active B and C block
BK: reduction depth for the active A and B panels
```

For a block update, the rough working set is:

```text
BM * BK elements from A
BK * BN elements from B
BM * BN elements from C
```

Assuming these blocks stay in cache during one block update, the work performed
by that update is:

```text
2 * BM * BN * BK floating-point operations
```

The rough arithmetic intensity is therefore:

```text
2 * BM * BN * BK
------------------------------------------ flops per byte
(BM * BK + BK * BN + BM * BN) * sizeof(T)
```

For the default `float` block size `64 x 64 x 64`:

```text
working set = (64*64 + 64*64 + 64*64) * 4 bytes
            = 49152 bytes

work        = 2 * 64 * 64 * 64
            = 524288 flops

intensity   = 524288 / 49152
            = 10.67 flops/byte
```

This is the main reason blocking helps: it increases the amount of computation
performed for each byte brought into cache. Without blocking, a simple loop
order can stream through large parts of `A` and `B` with little reuse before the
data is evicted. With blocking, a smaller `A` block, `B` block, and `C` block
are reused many times while they are still cache-resident.

This model is intentionally idealized. Real results also depend on cache
associativity, prefetching, TLB behavior, write allocation, register reuse,
compiler optimization, and whether the selected block sizes actually fit the
target cache.

The current implementation uses compile-time defaults:

```text
BM = 64
BN = 64
BK = 64
```

These are starting values, not tuned values. Later benchmarks should compare
rectangular alternatives such as `64 x 64 x 32` or `64 x 128 x 32`.

### `gemm_packed_b`

Adds packing for the active `B` panel:

```cpp
for (int jj = 0; jj < N; jj += BN)
  for (int kk = 0; kk < K; kk += BK)
    pack B[kk:kk+BK, jj:jj+BN] into Bpack

    for (int ii = 0; ii < M; ii += BM)
      update C[ii:ii+BM, jj:jj+BN]
      using A[ii:ii+BM, kk:kk+BK]
      and packed B
```

This uses block-level `j-k-i` order. The reason is that the same packed `B`
panel can be reused across many `ii` blocks. Packing has a copy cost, so it only
pays off when the packed panel is reused enough.

The current packed layout is compact row-major:

```cpp
Bpack[k * nc + j] = B[k * ldb + j];
```

This matches the scalar inner kernel, where `j` is the innermost loop and each
row of packed `B` is consumed contiguously. This is not a general conversion to
column-major. Packing means copying into the order the compute kernel wants.

This first packed-B implementation does not pad edge panels with zero. Edge
blocks are packed as their actual `kc x nc` size, and the kernel receives those
actual dimensions. Zero padding becomes more useful later when introducing a
fixed-size register micro-kernel.

### A Packing Variants

There are currently two A-packing experiments:

```text
gemm_packed_ab
  pack each A block immediately before it is used
  lower temporary memory footprint
  may repack the same A block when jj changes

gemm_packed_ab_prepack_a
  prepack all A block panels once before the main GEMM loop
  avoids repeated A packing across jj blocks
  uses a larger Apack_all buffer
```

Both variants still pack `B` panel-by-panel and reuse each packed `B` panel
across many `ii` blocks.

### `gemm_packed_ab`

Packs both operands using the first simple packed-AB strategy:

```cpp
for (int jj = 0; jj < N; jj += BN)
  for (int kk = 0; kk < K; kk += BK)
    pack B[kk:kk+BK, jj:jj+BN] into Bpack

    for (int ii = 0; ii < M; ii += BM)
      pack A[ii:ii+BM, kk:kk+BK] into Apack
      update C[ii:ii+BM, jj:jj+BN]
      using packed A and packed B
```

`Bpack` is still reused across many `ii` blocks. `Apack` is packed immediately
before use for each `ii` block. This means the same `A` block may be packed
again when `jj` changes, but the packed `A` data is hot when the compute kernel
uses it. This is the simplest bridge from packed panels to a future
micro-kernel.

The current packed layouts are compact row-major:

```cpp
Apack[i * kc + k] = A[i * lda + k];
Bpack[k * nc + j] = B[k * ldb + j];
```

The packed-AB kernel no longer depends on `lda` or `ldb` in the hot inner
compute loop. It only uses compact packed strides `kc` for `Apack` and `nc` for
`Bpack`, plus `ldc` for updating the original output matrix.

Like `gemm_packed_b`, this implementation does not pad edge panels with zero.
It packs actual `mc x kc` and `kc x nc` edge blocks and passes those actual
dimensions into the kernel.

### `gemm_packed_ab_register_blocked`

Adds a first scalar register-blocked micro-kernel on top of packed `A` and
packed `B`:

```cpp
for each packed A/B block
  for (int i = 0; i + 4 <= mc; i += 4)
    for (int j = 0; j + 4 <= nc; j += 4)
      compute one 4x4 C tile in scalar local variables
```

The key difference from `gemm_packed_ab` is how the hot block kernel updates
`C`. The older packed-AB kernel updates `C[i * ldc + j]` inside the innermost
loops. The register-blocked variant loads a `4 x 4` output tile into local
variables, accumulates across `kc` with explicit `std::fma` for floating-point
types, and stores the tile back once.

The `B` panel keeps the compact row-major layout used by `gemm_packed_ab`:

```cpp
Bpack[k * nc + j]
```

The `A` block is repacked into `4`-row micro-panels because the register-blocked
micro-kernel consumes four A values for the same `k` at once:

```cpp
ApackPanel[k * 4 + r]  // r = 0..3
```

For one `4 x kc` A micro-panel, memory is organized as:

```text
k=0: A0,0  A1,0  A2,0  A3,0
k=1: A0,1  A1,1  A2,1  A3,1
k=2: A0,2  A1,2  A2,2  A3,2
...
```

Full `4 x 4` tiles use the new micro-kernel. Edge rows or columns that do not
fit the `4 x 4` shape fall back to the scalar packed-AB block logic. This keeps
the implementation additive and makes the learning progression easy to compare
in benchmarks.

#### Outer-Product View

The `4 x 4` register block can be viewed as a sequence of tiny outer products.
For one fixed `k`, the micro-kernel loads four values from an `A` column slice
and four values from a `B` row slice:

```text
A slice:        B slice:

  a0              b0  b1  b2  b3
  a1
  a2
  a3
```

Those values form a `4 x 4` outer product:

```text
              b0        b1        b2        b3
          +---------+---------+---------+---------+
a0  ->    | a0 * b0 | a0 * b1 | a0 * b2 | a0 * b3 |
          +---------+---------+---------+---------+
a1  ->    | a1 * b0 | a1 * b1 | a1 * b2 | a1 * b3 |
          +---------+---------+---------+---------+
a2  ->    | a2 * b0 | a2 * b1 | a2 * b2 | a2 * b3 |
          +---------+---------+---------+---------+
a3  ->    | a3 * b0 | a3 * b1 | a3 * b2 | a3 * b3 |
          +---------+---------+---------+---------+
```

The micro-kernel accumulates that outer product into a `4 x 4` tile of `C`
held in scalar local variables:

```text
              b0        b1        b2        b3
          +---------+---------+---------+---------+
a0  ->    |   c00   |   c01   |   c02   |   c03   |
          +---------+---------+---------+---------+
a1  ->    |   c10   |   c11   |   c12   |   c13   |
          +---------+---------+---------+---------+
a2  ->    |   c20   |   c21   |   c22   |   c23   |
          +---------+---------+---------+---------+
a3  ->    |   c30   |   c31   |   c32   |   c33   |
          +---------+---------+---------+---------+
```

Each cell update is:

```cpp
cij = std::fma(ai, bj, cij);
```

So the full scalar register-blocked kernel is:

```cpp
load 4 x 4 C tile into c00 ... c33

for (int k = 0; k < kc; ++k) {
  load a0, a1, a2, a3
  load b0, b1, b2, b3

  c00 = fma(a0, b0, c00);  c01 = fma(a0, b1, c01);
  c02 = fma(a0, b2, c02);  c03 = fma(a0, b3, c03);

  c10 = fma(a1, b0, c10);  c11 = fma(a1, b1, c11);
  c12 = fma(a1, b2, c12);  c13 = fma(a1, b3, c13);

  c20 = fma(a2, b0, c20);  c21 = fma(a2, b1, c21);
  c22 = fma(a2, b2, c22);  c23 = fma(a2, b3, c23);

  c30 = fma(a3, b0, c30);  c31 = fma(a3, b1, c31);
  c32 = fma(a3, b2, c32);  c33 = fma(a3, b3, c33);
}

store c00 ... c33 back to C
```

This is still scalar register blocking. SIMD is the next step, where the same
outer-product idea is expressed with vector registers instead of individual
scalar variables.

### `gemm_packed_ab_register_blocked_avx2_float`

Adds a first SIMD micro-kernel for `float` using AVX2/FMA. It keeps the same
packed-AB outer loop as the scalar register-blocked variant, uses the same
`kc x 4` A micro-panel layout, and uses a `4 x 8` micro-kernel for full tiles:

```text
MR = 4 rows
NR = 8 columns
```

The `NR = 8` choice is specific to `float` on AVX2:

```text
256-bit AVX2 vector / 32-bit float = 8 floats per vector
```

For one output tile, the kernel keeps four vector registers for `C`:

```text
c0 = C row 0, columns j..j+7
c1 = C row 1, columns j..j+7
c2 = C row 2, columns j..j+7
c3 = C row 3, columns j..j+7
```

For each `k`, it loads one contiguous vector from packed `B` and broadcasts four
scalar values from one packed `A` micro-panel column:

```cpp
b  = load Bpack[k, j:j+8]
a0 = broadcast ApackPanel[k * 4 + 0]
a1 = broadcast ApackPanel[k * 4 + 1]
a2 = broadcast ApackPanel[k * 4 + 2]
a3 = broadcast ApackPanel[k * 4 + 3]

c0 = fmadd(a0, b, c0)
c1 = fmadd(a1, b, c1)
c2 = fmadd(a2, b, c2)
c3 = fmadd(a3, b, c3)
```

The AVX2 variant stores its temporary `Bpack` buffer with 32-byte alignment and
pads each packed B row to a multiple of `8` floats. This means every full-tile
B vector load is 32-byte aligned, even for edge `N` panels whose logical `nc`
is not a multiple of `8`.

The micro-kernel therefore uses aligned vector loads:

```cpp
b = _mm256_load_ps(Bpack + k * bpackStride);
```

Here `bpackStride` is the padded row width:

```cpp
bpackStride = round_up(nc, 8)
```

This is the SIMD version of the same outer-product idea. Instead of updating
sixteen scalar `C` variables for a `4 x 4` tile, it updates four vector
registers for a `4 x 8` tile. Edge rows or columns fall back to scalar logic
that understands the `kc x 4` A micro-panel layout and padded B row stride.

### `gemm_packed_ab_prepack_a`

Pre-packs all `A` block panels once before the main GEMM loop:

```cpp
for each kk block
  for each ii block
    pack A[ii:ii+BM, kk:kk+BK] into Apack_all

for (int jj = 0; jj < N; jj += BN)
  for (int kk = 0; kk < K; kk += BK)
    pack B[kk:kk+BK, jj:jj+BN] into Bpack

    for (int ii = 0; ii < M; ii += BM)
      update C[ii:ii+BM, jj:jj+BN]
      using prepacked A panel and packed B
```

This keeps the same block-level `j-k-i` compute order as `gemm_packed_ab`, but
removes repeated packing of the same `A[ii, kk]` panel when `jj` changes.

The tradeoff is memory footprint. Instead of a temporary `Apack` buffer of
`BM * BK` elements, this variant stores one packed slot for every `(kk, ii)`
panel:

```text
ceil(K / BK) * ceil(M / BM) * BM * BK elements
```

This can be close to the full size of `A`, plus edge-panel slack from fixed
panel slots. It is a useful experiment for measuring whether avoiding repeated
`A` packing outweighs the larger packed buffer.

## M, N, K vs. lda, ldb, ldc

`M`, `N`, and `K` describe the logical math problem:

```text
M: rows of A and C
N: columns of B and C
K: columns of A and rows of B
```

`lda`, `ldb`, and `ldc` describe the physical memory stride between adjacent
rows:

```text
lda: row stride of A, in elements
ldb: row stride of B, in elements
ldc: row stride of C, in elements
```

They are not redundant. For contiguous matrices, the leading dimensions usually
match the logical row width. For padded matrices, aligned allocations, or
submatrix views, the leading dimension can be larger than the logical width.

For example, if `A` is a `10 x 20` submatrix inside a larger row-major matrix
whose full row width is `100`, then:

```text
M = 10
K = 20
lda = 100
```

## Learning Path

The intended progression is:

1. Start with the reference row-major GEMM contract.
2. Compare `gemm_naive` against `gemm_ikj`.
3. Add cache blocking with `gemm_blocked`.
4. Add panel packing with `gemm_packed_b`.
5. Add packed A blocks with `gemm_packed_ab`.
6. Compare full A prepacking with `gemm_packed_ab_prepack_a`.
7. Add scalar register blocking with `gemm_packed_ab_register_blocked`.
8. Add SIMD register blocking with `gemm_packed_ab_register_blocked_avx2_float`.
9. Make every optimized kernel implement the same visible contract.

The visible algorithm should remain:

```text
C = alpha * A * B + beta * C
```

Optimization work should change how the computation is organized internally,
not what the public interface computes.
