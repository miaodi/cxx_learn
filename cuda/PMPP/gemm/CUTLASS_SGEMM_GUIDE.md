# CUTLASS SGEMM Tile-Shape Tuning Guide

This document explains how to read, configure, and reason about CUTLASS GEMM
kernel parameters for SIMT (scalar FMA) SGEMM on Hopper (H100).

## 1. The GEMM Hierarchy

CUTLASS decomposes `D = alpha * A * B + beta * C` into a 3-level tiling hierarchy:

```
                  Problem (M × N × K)
                         │
              ┌──────────┴──────────┐
              │  ThreadblockShape   │   ← one CTA (threadblock)
              │   <TbM, TbN, TbK>   │     computes TbM × TbN output
              └──────────┬──────────┘
                         │  tile the threadblock
              ┌──────────┴──────────┐
              │     WarpShape       │   ← one warp within the CTA
              │  <WpM, WpN, WpK>    │     computes WpM × WpN output
              └──────────┬──────────┘
                         │  tile the warp
              ┌──────────┴──────────┐
              │  InstructionShape   │   ← one HW instruction
              │   <1, 1, 1> SIMT    │     scalar FMA for OpClassSimt
              │   <16,8,8> Tensor   │      tensor core MMA for OpClassTensorOp
              └─────────────────────┘
```

## 2. Template Parameters of `cutlass::gemm::device::Gemm`

```cpp
cutlass::gemm::device::Gemm<
    ElementA,           // 1.  Data type of A                          (float)
    LayoutA,            // 2.  Memory layout of A                      (RowMajor / ColumnMajor)
    ElementB,           // 3.  Data type of B                          (float)
    LayoutB,            // 4.  Memory layout of B                      (RowMajor / ColumnMajor)
    ElementC,           // 5.  Data type of C and D                    (float)
    LayoutC,            // 6.  Memory layout of C and D                (RowMajor)
    ElementAccumulator, // 7.  Type for the MAC accumulator            (float, double)
    OperatorClass,      // 8.  OpClassSimt or OpClassTensorOp
    ArchTag,            // 9.  Target arch: Sm80, Sm90, …
    ThreadblockShape,   // 10. GemmShape<M,N,K> — output tile per CTA
    WarpShape,          // 11. GemmShape<M,N,K> — output tile per warp
    InstructionShape,   // 12. GemmShape<M,N,K> — per HW instruction
    EpilogueOp,         // 13. Epilogue functor (LinearCombination)
    ThreadblockSwizzle, // 14. CTA-to-tile mapping policy
    Stages              // 15. Pipeline depth (# shared-memory buffers)
>;
```

### 2.1 ElementA/B/C & Layouts (params 1–6)

These define the data types and whether matrices are stored row-major or
column-major.  All our benchmarks use `float` + `RowMajor`.

**RowMajor** means element `(i,j)` is at address `base + i * lda + j`.
CUTLASS internally pads shared-memory tiles to avoid bank conflicts; the
layout affects which padding scheme is legal.

### 2.2 ElementAccumulator (param 7)

The type used for the running sum in the MAC inner loop.  Using `float` for
SGEMM.  For mixed-precision (e.g., FP16 inputs) you'd use `float` here to
avoid precision loss.

### 2.3 OperatorClass (param 8)

| Value | Meaning |
|-------|---------|
| `OpClassSimt` | Scalar FMA instructions (`FFMA`). Each thread does 1 multiply-add per cycle. |
| `OpClassTensorOp` | Tensor Core MMA instructions. Each warp executes a matrix multiply on a small tile (e.g. 16×8×8). |

Our benchmarks use `OpClassSimt` to compare against hand-written SIMT kernels.
`OpClassTensorOp` with TF32 would be the natural next step to close the gap
with cuBLAS (which likely uses tensor cores).

### 2.4 ArchTag (param 9)

Selects which code path CUTLASS emits.  `Sm80` (Ampere) is the latest fully
supported SIMT code path.  CUTLASS runs `Sm80` kernels on Hopper (sm_90) via
forward compatibility — no performance penalty for SIMT.

### 2.5 ThreadblockShape (param 10) — **the most important knob**

`GemmShape<TbM, TbN, TbK>` defines the output tile computed by one CTA.

- **TbM × TbN** = output elements per CTA.
- **TbK** = depth of the K-dimension processed per "mainloop iteration"
  (one load from GMEM → shared memory, then multiply-accumulate).

**Derived quantities:**

| Metric | Formula | Why it matters |
|--------|---------|----------------|
| Grid blocks | `ceil(M/TbM) × ceil(N/TbN)` | Must be ≥ #SMs for full utilization |
| Smem per CTA | `Stages × (TbM×TbK + TbK×TbN) × sizeof(float)` | Limits occupancy |
| Arithmetic intensity | `2×TbM×TbN / [(TbM+TbN) × 4]` FLOP/byte | Higher → more compute-bound |

### 2.6 WarpShape (param 11)

`GemmShape<WpM, WpN, WpK>` defines the output tile per warp.

**Constraints:**
- `TbM / WpM` and `TbN / WpN` must be integers → gives warp grid dimensions.
- `Warps/CTA = (TbM/WpM) × (TbN/WpN)`.
- `Threads/CTA = Warps/CTA × 32`.
- For SIMT: `WpK` must equal `TbK` (warp processes full K-tile).

Within each warp, CUTLASS maps 32 threads to a 2D "thread tile" covering
WpM × WpN.  Each thread accumulates a small register sub-matrix.  Bigger warp
tiles → more registers per thread → potentially lower occupancy but better reuse.

### 2.7 InstructionShape (param 12)

For SIMT: always `GemmShape<1,1,1>` — one scalar FMA per thread per cycle.

For TensorOp: `GemmShape<16,8,8>` (Ampere), `GemmShape<16,8,16>` (Hopper TF32),
etc.  This tells CUTLASS which `mma` PTX instruction to emit.

### 2.8 EpilogueOp (param 13)

The functor applied after the accumulator is complete:

```cpp
cutlass::epilogue::thread::LinearCombination<
    float,   // ElementOutput
    1,       // Elements per access (1 for SIMT, 4 or 8 for vectorized)
    float,   // ElementAccumulator
    float    // ElementCompute (for alpha/beta arithmetic)
>
```

Computes `D[i] = alpha * accumulator[i] + beta * C[i]`.

### 2.9 ThreadblockSwizzle (param 14)

Controls the mapping from `blockIdx` to output tile coordinates.

| Policy | Behavior |
|--------|----------|
| `GemmIdentityThreadblockSwizzle<>` | Linear (row-major) mapping. Simple, predictable. |
| `GemmIdentityThreadblockSwizzle<8>` | Groups of 8 CTAs in a column before moving right. Can improve L2 locality for tall-skinny problems. |
| `GemmHorizontalThreadblockSwizzle` | Column-major mapping. |

For square matrices, identity swizzle is usually fine.

### 2.10 Stages (param 15) — pipeline depth

Number of shared-memory buffers for double/multi-buffering the GMEM→SMEM loads.

- **2 stages** (double buffering): While computing on buffer 0, load into buffer 1.
  Smem = 2 × tile_size.
- **4 stages**: 4× the smem, hides more GMEM latency.  Useful when GMEM is the
  bottleneck and the SM has enough shared memory.
- **More stages** ≠ always better: each extra buffer costs shared memory, which
  can reduce the number of concurrent CTAs per SM.

## 3. How to Pick Good Parameters

### Step 1: Choose grid size (ThreadblockShape M×N)

For a problem of size M×N on a GPU with S SMs:

```
blocks = ceil(M/TbM) × ceil(N/TbN)
```

You want `blocks ≥ S` and ideally `blocks ≈ k×S` for integer k (full "waves").
H100 has 132 SMs.

| TbM×TbN | Blocks for 1024² | Waves on 132 SMs |
|---------|------------------|-------------------|
| 128×128 | 64 | 0.48 ← under-filled |
| 64×128 | 128 | 0.97 ← nearly 1 full wave |
| 64×64 | 256 | 1.94 ← ~2 waves |
| 32×128 | 256 | 1.94 |

### Step 2: Choose warp shape

Warps/CTA should be 4 (128 threads) or 8 (256 threads).
- 128 threads: lower register pressure, more CTAs per SM.
- 256 threads: can compute larger tiles, but fewer concurrent CTAs.

The warp shape determines each thread's register tile.  Wider N tiles
coalesce better for RowMajor output.

### Step 3: Choose K-tile and stages

- `TbK = 8` is the standard for FP32 SIMT.  Smaller K-tiles mean more
  mainloop iterations but less smem.
- Start with 2 stages.  Try 4 if profiling shows GMEM latency is dominant.

### Step 4: Measure!

Theory gives you a starting point.  The final answer comes from benchmarking.

## 4. Benchmark Results on H100 (1024×1024 SGEMM)

From our runs:

| Config | ThreadblockShape | Grid | Threads/CTA | Smem | Time (ms) | GFLOP/s |
|--------|-----------------|------|-------------|------|-----------|---------|
| Default | (auto) | (auto) | (auto) | (auto) | 0.102 | 21.1T |
| 128×128×8 | 128×128×8 | 64 | 256 | 16K | 0.102 | 21.0T |
| 64×64×8 | 64×64×8 | 256 | 128 | 8K | 0.072 | 29.8T |
| **64×128×8** | **64×128×8** | **128** | **128** | **12K** | **0.063** | **33.9T** |
| 128×128×8 4stg | 128×128×8 | 64 | 256 | 32K | 0.107 | 20.1T |
| cuBLAS | — | — | — | — | 0.065 | 32.8T |

### Why 64×128×8 wins

1. **128 blocks ≈ 1 wave on 132 SMs** — nearly all SMs active, minimal tail effect.
2. **128 threads/CTA** — moderate register pressure → good occupancy.
3. **12 KiB smem** — small enough for multiple CTAs per SM.
4. **21.3 FLOP/byte arithmetic intensity** — balanced between compute and memory.
5. **Wide N (128) with RowMajor** — contiguous writes to D across the N dimension.

### Why 128×128×8 is slower

Only 64 blocks for 132 SMs — half the SMs sit idle.  The higher arithmetic
intensity (32 FLOP/byte) doesn't compensate for the parallelism loss.

### Why 4-stage is slower for 128×128×8

32 KiB smem/CTA × 64 blocks.  With only 64 blocks and 256 threads/CTA,
the extra smem reduces occupancy without enough blocks to hide it.

## 5. Next Steps for Further Optimization

1. **Tensor Cores (TF32)**: Replace `OpClassSimt` with `OpClassTensorOp` and
   `InstructionShape<16,8,8>`.  This accesses the tensor cores that cuBLAS uses.
   Expected ~2-4× over SIMT.

2. **Larger matrices**: At 4096×4096, the 128×128 tile has 1024 blocks — enough
   to saturate H100.  The optimal tile shape shifts with problem size.

3. **Swizzle policies**: Try `GemmIdentityThreadblockSwizzle<8>` for better L2
   reuse patterns.

4. **Split-K**: For small M×N but large K, split the K dimension across multiple
   CTAs and reduce.  Adds a workspace buffer but increases parallelism.
