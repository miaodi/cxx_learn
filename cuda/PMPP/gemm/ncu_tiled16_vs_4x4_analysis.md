# Nsight Compute Tiled16 vs Tiled16_4x4 Analysis

This note compares:

```text
release/cuda/PMPP/gemm/ncu_sgemm_tiled16.ncu-rep
release/cuda/PMPP/gemm/ncu_sgemm_tiled16_4x4.ncu-rep
```

The reports were collected on the same RTX 4070 with Nsight Compute `2026.1.1`
and CUDA `13.2`.

Important caveat: the baseline report's kernel name is
`sgemm_tiled_kernel<16>`, while the `4x4` report's kernel name is
`sgemm_tiled_thread_tile_kernel<16, 4, 16>`. This file compares the profile
reports as collected. If `sgemm_tiled_16` is now routed to
`sgemm_tiled_thread_tile_kernel<16, 1, 16>`, re-profile it for a perfectly
current baseline.

## One-Line Result

`Tiled16_4x4` is faster because each thread computes a `4x4` patch of `C`,
which raises data reuse and amortizes shared/global memory work over much more
math. It launches `1/16` as many blocks and threads, executes about `36%` as
many instructions, and reaches about `4.3x` the FFMA issue rate of the baseline.

The cost is higher register pressure, more shared memory per block, lower
occupancy, less total parallelism at `1024x1024`, and measurable shared-store
bank conflicts.

## Benchmark And NCU Timing

Your Google Benchmark result:

| Variant | Benchmark time | Benchmark throughput |
| --- | ---: | ---: |
| `Tiled16` | 0.855 ms | 2.512 TFLOP/s |
| `Tiled16_4x4` | 0.211 ms | 10.189 TFLOP/s |

Benchmark speedup: `4.05x`.

NCU reports:

| Variant | NCU duration | Derived throughput |
| --- | ---: | ---: |
| `Tiled16` | 933.50 us | 2.300 TFLOP/s |
| `Tiled16_4x4` | 223.78 us | 9.597 TFLOP/s |

NCU speedup: `4.17x`.

The NCU and benchmark measurements agree on the main conclusion: `4x4` is
roughly `4x` faster.

## Launch Shape

| Metric | `Tiled16` | `Tiled16_4x4` |
| --- | ---: | ---: |
| Kernel | `sgemm_tiled_kernel<16>` | `sgemm_tiled_thread_tile_kernel<16, 4, 16>` |
| Block size | 256 threads | 256 threads |
| Output tile/block | `16x16` | `64x64` |
| Accumulators/thread | 1 | 16 |
| Grid blocks | 4096 | 256 |
| Threads launched | 1048576 | 65536 |
| Waves/SM | 14.84 | 1.39 |
| Registers/thread | 37 | 59 |
| Shared memory/block | 2048 B static | 8192 B dynamic |

The reduced grid is not an accident: `4x4` does more work per block. For the
same `1024x1024` output matrix, it needs only `16x16 = 256` blocks instead of
`64x64 = 4096` blocks.

## Why 4x4 Is More Efficient

The baseline stages a `16x16` tile of `A` and a `16x16` tile of `B` for each K
step, then computes one `C` element per thread. Per K tile, that is:

```text
Loaded operands: 16*16 + 16*16 = 512 floats
Work:            16*16*16 FMA = 4096 FMA = 8192 FLOP
Reuse:           16 FLOP per loaded float
```

The `4x4` kernel stages a `64x16` tile of `A` and a `16x64` tile of `B`, then
each thread computes 16 output elements:

```text
Loaded operands: 64*16 + 16*64 = 2048 floats
Work:            64*64*16 FMA = 65536 FMA = 131072 FLOP
Reuse:           64 FLOP per loaded float
```

So the per-tile arithmetic intensity is about `4x` higher. That matches the
measured speedup closely.

NCU confirms this:

| Metric | `Tiled16` | `Tiled16_4x4` | Meaning |
| --- | ---: | ---: | --- |
| Instructions executed | 129859584 | 46477312 | `4x4` has much less instruction overhead |
| FFMA thread inst/cycle | 466.26 | 1996.43 | `4x4` feeds the FMA pipe much better |
| Issue active | 30.62% | 46.94% | schedulers issue useful work more often |
| Shared load wavefronts | 50335848 | 12583166 | `4x4` reduces shared-load traffic to about 25% |
| Total shared wavefronts | 54525952 | 15728640 | shared-memory work drops to about 29% |
| MIO throttle per issue-active | 18.84 | 3.42 | much less load/store-pipe pressure per issued instruction |
| Barrier per issue-active | 5.68 | 1.80 | synchronization cost is amortized over more math |
| Long scoreboard per issue-active | 5.63 | 1.47 | less waiting on memory dependencies |

This is the main reason `4x4` wins: it does not make each individual memory
operation magically faster; it gets much more math from each staged tile and
uses fewer total instructions to complete the same GEMM.

## Bottleneck Classification

Neither kernel is primarily DRAM-bandwidth-bound.

| Metric | `Tiled16` | `Tiled16_4x4` |
| --- | ---: | ---: |
| DRAM throughput | 3.73% | 15.12% |
| DRAM bandwidth | 18.33 GB/s | 74.33 GB/s |
| L1/TEX throughput | 96.08% | 84.04% |
| SM throughput | 95.12% | 51.36% |

The baseline is dominated by SM/L1/load-store pressure. The very high L1/TEX
and LSU utilization, combined with large MIO-throttle stalls, says it is
spending too much scheduler bandwidth on memory movement and shared-memory
loads relative to useful FMAs.

The `4x4` kernel shifts the balance toward arithmetic. It still is not near
peak FP32 throughput, but it uses the FMA pipe much more effectively and
reduces memory-pipeline stalls substantially.

## Costs And Warnings In 4x4

`4x4` is better, but it is not free.

| Metric | `Tiled16` | `Tiled16_4x4` |
| --- | ---: | ---: |
| Active warps/SM | 46.76 | 25.43 |
| Active warp occupancy | 97.42% | 52.99% |
| Eligible warps/scheduler | 1.37 | 1.75 |
| Register occupancy limit | 6 blocks/SM | 4 blocks/SM |
| Shared-memory occupancy limit | 21 blocks/SM | 7 blocks/SM |
| Shared bank conflicts | 0 | 2097152 |
| Excess global sectors | 0 | 7077888 |
| Short scoreboard per issue-active | 0.39 | 1.12 |

Lower occupancy is acceptable here because each warp has more independent work
and more reuse. The higher eligible-warps/scheduler metric shows that the lower
active-warp count is not hurting as much as it might look from occupancy alone.

The real concerns are:

1. **Shared-store bank conflicts:** NCU reports `2097152` shared-memory bank
   conflicts, all on shared stores.
2. **Uncoalesced/global-sector inefficiency:** theoretical global sectors are
   `11534336` versus an ideal `4456448`, so NCU reports `7077888` excessive
   global sectors.
3. **Register pressure:** `59` registers/thread is still reasonable, but it
   limits resident blocks to 4/SM. Larger thread tiles such as `8x8` can cross
   the point where register pressure and grid underfill dominate.

## Why 8x8 Is Slower Than 4x4

Your benchmark shows:

```text
Tiled16_4x4: 0.211 ms
Tiled16_8x8: 0.276 ms
```

This is expected at `1024x1024`. `8x8` computes a `128x128` output tile per
block, so it launches only `8x8 = 64` blocks. On a 46-SM RTX 4070, that is only
about 1.4 blocks per SM total. It also needs many more accumulators per thread.
The extra reuse is real, but at this problem size the loss of parallelism and
register pressure outweigh it.

## What To Do Next

1. **Re-profile the current `Tiled16` baseline.** Since `sgemm_tiled_16` was
   changed to dispatch through `sgemm_tiled_thread_tile_impl<16, 1, 16>`, collect
   a fresh `ncu_sgemm_tiled16_1x1.ncu-rep` and compare that against `4x4`.

2. **Fix the `4x4` shared-store bank conflicts.** Try padding the leading
   dimension of `tileA` or `tileB` in shared memory. For example, allocate one
   tile with a stride of `KTile + 1` or `OutputCols + 1`, then re-check:

   ```text
   l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
   l1tex__data_pipe_lsu_wavefronts_mem_shared.sum
   gpu__time_duration.sum
   ```

   This is the smallest high-confidence next experiment because NCU identifies
   a concrete conflict count in the faster kernel.

3. **Try asymmetric thread tiles.** `4x8` or `8x4` may keep more reuse than
   `4x4` while avoiding some of the grid-underfill/register-pressure problems
   of `8x8`. For `1024x1024`, these would launch `128` blocks, less than `4x4`
   but more than `8x8`.

4. **Profile larger matrices.** Test `2048` and `4096`. Larger matrices give
   bigger tiles enough block-level parallelism, so the ranking of `4x4`, `8x8`,
   and asymmetric variants may change.

5. **Consider increasing K tile only after bank conflicts are understood.**
   `KTile=32` or `64` can improve reuse and reduce loop overhead, but it also
   increases shared memory and can hurt occupancy. Compare with NCU rather than
   assuming bigger K tiles are better.

6. **Use cuBLAS as the ceiling, not the target implementation style.** cuBLAS
   can use architecture-specific tiling, pipelining, vectorized memory
   operations, and possibly tensor cores depending on math mode. Your SIMT FP32
   kernels are good for learning the optimization path, but they are not
   expected to match cuBLAS just by changing tile sizes.

## Validation Plan

For each new variant, collect both benchmark and NCU data:

```bash
./PMPP_gemm_bench --benchmark_filter='SGEMM/Tiled16_4x4/1024' \
    --benchmark_min_time=1s

ncu --set full --force-overwrite -o ncu_sgemm_tiled16_4x4 \
    ./PMPP_gemm_bench --benchmark_filter='SGEMM/Tiled16_4x4/' \
    --benchmark_min_time=1x
```

Judge changes first by benchmark time, then use NCU to explain the result. For
the next iteration, the most important NCU counters are kernel duration, FFMA
issue rate, shared bank conflicts, shared wavefronts, global excessive sectors,
registers/thread, active warps/SM, and eligible warps/scheduler.
