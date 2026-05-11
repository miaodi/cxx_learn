# Nsight Compute Tiled16 Tile-A Layout Analysis

This note compares two Nsight Compute reports for the same kernel entry point,
`sgemm_tiled_kernel<16>`, collected from:

```text
release/cuda/PMPP/gemm/ncu_sgemm_tiled16.ncu-rep
release/cuda/PMPP/gemm/ncu_sgemm_tiled16_trans.ncu-rep
```

Both reports profile the same benchmark filter:

```text
./PMPP_gemm_bench --benchmark_filter=SGEMM/Tiled16/ --benchmark_min_time=1x
```

The difference between the two runs is the shared-memory layout of `tileA`:

| Report | Meaning |
| --- | --- |
| `ncu_sgemm_tiled16.ncu-rep` | Baseline shared tile layout, storing `A` as loaded. |
| `ncu_sgemm_tiled16_trans.ncu-rep` | `tileA` manually stored in transposed shared-memory form. |

## Profile Context

| Item | Value |
| --- | ---: |
| GPU | NVIDIA GeForce RTX 4070 |
| Compute capability | 8.9 |
| SM count | 46 |
| CUDA version | 13.2 |
| Nsight Compute | 2026.1.1 |
| Grid size | 4096 blocks |
| Block size | 256 threads |
| Threads launched | 1048576 |
| Waves per SM | 14.84 |
| Static shared memory/block | 2048 bytes |

The profile is for a `1024 x 1024 x 1024` SGEMM benchmark. Derived throughput
below uses `2 * 1024^3` floating-point operations and the NCU-reported kernel
duration. Treat this as profile-context throughput; normal benchmark timing can
differ from NCU timing.

## Summary

The transposed `tileA` layout is slightly slower in this profile. The baseline
run takes `933.50 us`; the transposed-`tileA` run takes `962.02 us`, a `3.05%`
increase in kernel duration.

The reports do not show a reduction in global-memory traffic, shared-memory
wavefronts, or shared-memory bank conflicts from transposing `tileA`. The main
measured differences are small: the transposed version uses one fewer register
per thread, but has lower achieved active warps, lower issue activity, lower
FFMA issue rate, and a higher short-scoreboard stall sample count.

## Extracted Metrics

| Metric | Baseline `tileA` | Transposed `tileA` | Delta |
| --- | ---: | ---: | ---: |
| Kernel duration | 933.50 us | 962.02 us | +3.05% |
| Derived throughput | 2.300 TFLOP/s | 2.232 TFLOP/s | -2.96% |
| Elapsed cycles | 2305988 | 2377808 | +3.11% |
| SM throughput | 95.12% | 92.24% | -2.88 pp |
| L1/TEX throughput | 96.08% | 93.16% | -2.92 pp |
| L2 throughput | 24.48% | 23.70% | -0.78 pp |
| DRAM throughput | 3.73% | 3.65% | -0.08 pp |
| DRAM bandwidth | 18.33 GB/s | 17.92 GB/s | -2.21% |
| Registers/thread | 37 | 36 | -1 |
| Active warps/SM | 46.76 | 45.36 | -2.99% |
| Eligible warps/scheduler | 1.37 | 1.37 | no change |
| Issue active | 30.62% | 29.70% | -0.92 pp |
| FFMA thread instructions/cycle | 466.26 | 452.15 | -3.03% |
| Instructions executed | 129859584 | 129892352 | +0.03% |

## Memory Behavior

| Metric | Baseline `tileA` | Transposed `tileA` | Observation |
| --- | ---: | ---: | --- |
| Global load sectors | 16908288 | 16908288 | Same |
| Global store sectors | 131072 | 131072 | Same |
| Global load requests | 4227072 | 4227072 | Same |
| Global store requests | 32768 | 32768 | Same |
| L2 theoretical global sectors | 17039360 | 17039360 | Same |
| L2 ideal global sectors | 17039360 | 17039360 | Same |
| Excess global sectors | 0 | 0 | Same |
| Shared wavefronts | 54525952 | 54525952 | Same |
| Ideal shared wavefronts | 54525952 | 54525952 | Same |
| Excess shared wavefronts | 0 | 0 | Same |
| Shared load wavefronts | 50335848 | 50335788 | Same within noise |
| Shared store wavefronts | 4194304 | 4194304 | Same |
| Shared bank conflicts | 0 | 0 | Same |
| L1/TEX hit rate | 5.05% | 5.10% | Effectively same |
| L2 hit rate | 97.48% | 97.42% | Effectively same |

The layout change does not improve coalescing or reduce the amount of shared
memory work measured by NCU. Global memory is not the bottleneck: DRAM
throughput is only about `3.7%` of peak, while L1/TEX and SM-side LSU activity
are near the top of the Speed of Light breakdown.

## Stall Signals

The dominant stall category in both profiles is MIO throttle, which is
consistent with pressure around shared-memory/load-store issue paths rather
than HBM bandwidth.

| Stall metric | Baseline `tileA` | Transposed `tileA` |
| --- | ---: | ---: |
| MIO throttle per issue-active | 18.84 | 18.71 |
| Long scoreboard per issue-active | 5.63 | 5.64 |
| Barrier per issue-active | 5.68 | 5.67 |
| Not selected per issue-active | 3.45 | 3.42 |
| Short scoreboard per issue-active | 0.39 | 0.48 |
| MIO throttle not-issued samples | 41521 | 41676 |
| Long scoreboard not-issued samples | 10784 | 10754 |
| Barrier not-issued samples | 10155 | 10263 |
| Short scoreboard not-issued samples | 761 | 1014 |

The transposed variant does not change the main stall mix. The notable negative
change is the short-scoreboard not-issued sample count, which rises from `761`
to `1014`. This is not the largest stall category, but it matches the slight
drop in FFMA issue rate and total runtime.

## Interpretation

This kernel is not DRAM-bound and it is not reaching FP32 arithmetic peak. It is
mostly limited by SM-side load/store and shared-memory instruction pressure.
NCU's Speed of Light section reports high SM/L1 utilization, but the roofline
FFMA rate is only about `466` thread instructions/cycle for the baseline and
`452` for the transposed version, versus a theoretical `5888`.

Transposing `tileA` in shared memory does not help this specific `16 x 16`
one-output-per-thread kernel because the measured transaction shape remains the
same:

1. Global load/store sectors are unchanged.
2. Shared-memory wavefronts are unchanged.
3. NCU reports no shared-memory bank conflicts in either run.
4. Occupancy is already high enough to launch many waves per SM.

The transposed layout also changes the compiler's instruction/register shape
slightly. It saves one register per thread, but this does not raise occupancy
or eligible warps in a useful way. The measured result is instead a small drop
in issue activity and FFMA rate.

## Conclusion

For `sgemm_tiled_kernel<16>` at the profiled `1024` problem size, manually
storing `tileA` transposed in shared memory is not beneficial. The baseline
layout is about `3%` faster in the NCU report, and the memory counters show no
reduction in global traffic, shared traffic, or bank conflicts.

The next useful optimization target is not this `tileA` orientation. It is
reducing the amount of shared-memory/load-store instruction pressure per
floating-point operation, for example by using register blocking so each thread
computes multiple `C` elements, or by changing the inner loop and shared layout
only if a new profile shows fewer shared-memory wavefronts or lower MIO
throttle.

## Validation Commands

Export comparable details from the existing reports:

```bash
ncu --import release/cuda/PMPP/gemm/ncu_sgemm_tiled16.ncu-rep \
    --page details --print-details all --print-metric-name label-name

ncu --import release/cuda/PMPP/gemm/ncu_sgemm_tiled16_trans.ncu-rep \
    --page details --print-details all --print-metric-name label-name
```

Re-profile both variants under the same command line if the kernel source is
changed:

```bash
ncu --set full --force-overwrite -o ncu_sgemm_tiled16 \
    ./PMPP_gemm_bench --benchmark_filter=SGEMM/Tiled16/ --benchmark_min_time=1x
```
