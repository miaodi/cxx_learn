# GEMM Learning Workspace

This directory is a workspace for learning how GEMM implementations are built
and optimized.

GEMM means general matrix multiplication. The standard operation is:

```text
C = alpha * A * B + beta * C
```

The active kernels in `gemm.hpp` are `gemm_naive`, `gemm_ikj`,
`gemm_blocked`, `gemm_packed_b`, `gemm_packed_ab`, and
`gemm_packed_ab_prepack_a`. They all use this row-major BLAS-style contract.

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
void gemm_packed_ab_prepack_a(int M, int N, int K,
                              T alpha,
                              const T *A, int lda,
                              const T *B, int ldb,
                              T beta,
                              T *C, int ldc);
```

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
7. Make every optimized kernel implement the same visible contract.
8. Add register blocking and micro-kernels.
9. Add SIMD once the scalar micro-kernel shape is clear.

The visible algorithm should remain:

```text
C = alpha * A * B + beta * C
```

Optimization work should change how the computation is organized internally,
not what the public interface computes.
