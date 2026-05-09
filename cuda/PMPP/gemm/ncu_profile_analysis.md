# Nsight Compute SGEMM Profile Analysis

This note explains how to read the three Nsight Compute CSV exports for the
current register-blocked CUDA SGEMM kernels.

Profile inputs:

```text
release/cuda/PMPP/gemm/ncu_sgemm_2x2_details.csv
release/cuda/PMPP/gemm/ncu_sgemm_4x4_details.csv
release/cuda/PMPP/gemm/ncu_sgemm_8x8_details.csv
```

The profiles were collected for the fixed benchmark case:

```text
M = 1024
N = 1024
K = 1024
```

The three kernels use the same `16x16 = 256` CUDA threads per block. The
difference is how many `C` elements each thread computes:

| Variant | Output tile per block | Accumulators/thread | Grid blocks for 1024x1024 C |
| --- | ---: | ---: | ---: |
| `2x2` | `32 x 32` | 4 | 1024 |
| `4x4` | `64 x 64` | 16 | 256 |
| `8x8` | `128 x 128` | 64 | 64 |

## One-Line Result

For this `1024` matrix size, `4x4` is the best point among the three profiled
kernels. `2x2` has much higher occupancy but less reuse per thread. `8x8` does
more work per thread, but it uses many more registers, launches too few blocks
to fill an H100, and shows worse shared-memory conflict signals.

## Summary Table

The throughput below is computed from the NCU-reported kernel duration using
`2 * 1024^3` floating-point operations. Treat it as profile-context throughput,
not a replacement for the Google Benchmark result. NCU may replay kernels and
collect hardware counters, so its timings can differ from normal benchmark
timings.

| Metric | `2x2` | `4x4` | `8x8` |
| --- | ---: | ---: | ---: |
| NCU duration | 190.78 us | 135.14 us | 267.55 us |
| Derived throughput | 11.26 TFLOP/s | 15.89 TFLOP/s | 8.03 TFLOP/s |
| Grid blocks | 1024 | 256 | 64 |
| Threads launched | 262144 | 65536 | 16384 |
| Waves per SM | 0.97 | 0.48 | 0.24 |
| Registers/thread | 32 | 62 | 128 |
| Static shared memory/block | 4.10 KiB | 8.19 KiB | 16.38 KiB |
| Theoretical occupancy | 100% | 50% | 25% |
| Achieved occupancy | 75.62% | 24.32% | 12.47% |
| Theoretical active warps/SM | 64 | 32 | 16 |
| Achieved active warps/SM | 48.40 | 15.57 | 7.98 |
| Compute throughput | 42.58% | 44.68% | 20.35% |
| L1/TEX throughput | 89.65% | 76.66% | 84.09% |
| DRAM throughput | 1.97% | 2.78% | 1.41% |
| L1/TEX hit rate | 35.67% | 63.42% | 80.36% |
| L2 hit rate | 89.16% | 81.47% | 71.25% |
| Issued warp/scheduler | 0.43 | 0.48 | 0.43 |
| Eligible warps/scheduler | 1.36 | 0.94 | 0.66 |
| No eligible warp cycles | 57.26% | 51.59% | 57.41% |

## How To Read The NCU Sections

Start with `GPU Speed Of Light Throughput`. This is the fastest way to see
whether the kernel is mostly limited by compute, memory hierarchy, or launch
shape. In these profiles, DRAM throughput is tiny, around `1-3%`, while L1/TEX
throughput is much higher. That means the kernels are not HBM bandwidth-bound.
The pressure is closer to the SM/L1/shared-memory/register part of the machine.

Then check `Launch Statistics`. For `8x8`, this section is the first major red
flag:

```text
Grid Size = 64
# SMs = 132
Waves Per SM = 0.24
```

Only 64 thread blocks are launched for the whole GEMM, but the H100 has 132 SMs.
That means many SMs cannot receive even one block. This is not just an occupancy
issue inside one SM; it is a whole-GPU underfill issue.

Next check `Occupancy`. Occupancy is not the goal by itself, but it tells you
whether the kernel has enough resident warps to hide latency.

```text
2x2: 32 registers/thread  -> 100% theoretical occupancy
4x4: 62 registers/thread  ->  50% theoretical occupancy
8x8: 128 registers/thread ->  25% theoretical occupancy
```

The important lesson is that lower occupancy can still win. `4x4` is faster
than `2x2` because each loaded tile feeds more arithmetic. But `8x8` goes too
far for this matrix size: it has fewer blocks, fewer resident warps, and much
larger per-thread state.

Then inspect `Scheduler Statistics` and `Warp State Statistics`. These sections
answer: "Were warps ready to issue instructions?"

| Metric | Meaning |
| --- | --- |
| `Issued Warp Per Scheduler` | How often each scheduler actually issued a warp instruction. |
| `Eligible Warps Per Scheduler` | Warps that were resident and ready to issue. |
| `No Eligible` | Cycles where the scheduler had no ready warp. |
| `Warp Cycles Per Issued Instruction` | Average cycles between issued instructions for a warp. |

`4x4` has the best duration, but it still only issues about one warp every
`2.1` cycles per scheduler. NCU reports that many stalls are short scoreboard
stalls, commonly caused by shared-memory or other MIO dependencies. That points
to shared-memory layout and instruction scheduling as useful next places to
look.

Finally inspect `Memory Workload Analysis` and `Source Counters`. These sections
flag coalescing and shared-memory bank conflicts.

## Coalescing And Shared-Memory Warnings

NCU reports uncoalesced global accesses for all three kernels:

| Variant | Excessive global sectors | Share of sectors |
| --- | ---: | ---: |
| `2x2` | 4456448 | 34% |
| `4x4` | 7077888 | 61% |
| `8x8` | 9175040 | 80% |

This warning should be taken seriously, but it must be interpreted with the
kernel shape. As each thread computes a larger rectangular patch, the final
stores and some loads become less naturally contiguous at the warp level. `4x4`
still wins because the extra arithmetic reuse pays for the less ideal access
pattern. `8x8` shows the warning becoming severe enough that it combines with
register pressure and grid underfill.

NCU also reports shared-memory conflicts:

| Variant | Shared-memory warning |
| --- | --- |
| `2x2` | Shared stores average a `2.0`-way bank conflict. |
| `4x4` | Shared stores average a `3.1`-way bank conflict. |
| `8x8` | Shared loads average a `3.2`-way conflict; shared stores average a `5.1`-way conflict. |

This is a strong sign that the next implementation experiment should look at
shared-memory layout. Padding one shared-memory dimension, changing the shared
tile orientation, or separating the load layout from the compute layout can
reduce bank conflicts.

## Why 4x4 Is Best Here

`2x2` has excellent occupancy, but each thread only computes four outputs. The
kernel launches many blocks and many threads, executes more total instructions
than `4x4`, and still spends many cycles waiting on shared-memory dependencies.

`4x4` increases arithmetic reuse without yet destroying the launch. It uses 62
registers per thread and cuts theoretical occupancy to 50%, but it also reduces
thread count and instruction count while raising useful work per shared tile.
For this profile, that tradeoff wins.

`8x8` crosses the limit for this `1024` case. It launches only 64 blocks on a
132-SM GPU, uses 128 registers per thread, reaches only 12.47% achieved
occupancy, and has strong shared-memory conflict warnings. The larger per-thread
tile is not bad in principle, but at this matrix size it does not expose enough
parallel blocks to keep the whole GPU busy.

## What To Try Next

1. Profile `4x4` and `8x8` at larger matrix sizes such as `2048` or `4096`.
   The `8x8` variant may look better when the grid has enough blocks to fill
   the GPU.
2. Add a shared-memory padding experiment. For example, try padding the leading
   dimension of one staged tile by one float and check whether NCU's shared bank
   conflict warnings drop.
3. Check for register spills when pushing beyond `4x4`. Useful signs are high
   register count, local memory traffic, or `ptxas` spill messages.
4. Try asymmetric per-thread tiles such as `4x8` or `8x4`. These may preserve
   more reuse than `4x4` without reducing the block count as much as `8x8`.
5. Keep cuBLAS as the ceiling. The hand-written kernels are still SIMT FP32
   learning kernels; cuBLAS may use tensor cores, more advanced tiling, deeper
   pipelining, and architecture-specific scheduling.

## Useful Commands

Generate a details CSV from an existing `.ncu-rep` file:

```bash
ncu --import ncu_sgemm_4x4.ncu-rep --page details --csv > ncu_sgemm_4x4_details.csv
```

Collect only the sections used most in this analysis:

```bash
ncu --target-processes all \
    --section SpeedOfLight \
    --section LaunchStats \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    --section MemoryWorkloadAnalysis \
    --section SourceCounters \
    --kernel-name regex:sgemm_tiled_4x4_kernel \
    -o ncu_sgemm_4x4 \
    ./PMPP_gemm_bench \
    --benchmark_filter='SGEMM/Tiled16_4x4/1024' \
    --benchmark_min_time=1x
```

For clean timing, use Google Benchmark without NCU. For diagnosis, use NCU and
focus on relative bottlenecks rather than absolute runtime.
