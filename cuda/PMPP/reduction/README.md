# PMPP Reduction

## Concept

This example implements a progression of PMPP-style sum reductions:

- `simple_stride_sum`: interleaved addressing.
- `sequential_addressing_sum`: sequential addressing with contiguous active
  threads.
- `coarsened_shared_memory_sum`: thread coarsening plus shared-memory block
  reduction.
- `coarsened_shared_memory_sum_optimized`: the same coarsened shared-memory
  identity code with an unrolled load loop and warp-shuffle tail reduction.
- `thrust_reference_sum`: a Thrust reference implementation for comparison.

## What To Run

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=ON
cmake --build build-cuda --target PMPP_reduction_test PMPP_reduction_bench -j
ctest --test-dir build-cuda -R SimpleStrideSumTest --output-on-failure
./build-cuda/cuda/PMPP/reduction/PMPP_reduction_bench
```

## What To Look For

The GPU results should match the CPU sum within floating-point tolerance. The
benchmark assumes the input already lives in device memory: it allocates and
copies input once in fixture setup, then times only the reduction calls.

The custom kernels preserve the input like Thrust. They read the device input and
write intermediate values to reduction workspace buffers. All GPU paths still
copy the final scalar back to the host.

## Why It Happens

The simple stride version repeatedly doubles the stride:

```text
stride = 1, 2, 4, ...
thread tid adds input[tid + stride] when tid % (2 * stride) == 0
```

The sequential-addressing version keeps the same global-memory reduction idea,
but makes active threads contiguous at each step:

```text
first 256 elements add the second 256 elements
first 128 elements add the second 128 elements
first 64 elements add the second 64 elements
...
```

That second pattern reduces branch divergence and gives a cleaner memory-access
pattern inside each block. The first two variants still keep intermediate values
in global memory, which is much more expensive than shared memory.

The coarsened shared-memory version assigns each block the same `256` shared
slots, one per thread. Each thread first accumulates `K = 4` global-memory values
into a register and stores one partial sum to shared memory:

```text
shared[tid] = input[block_start + tid]
            + input[block_start + 256 + tid]
            + input[block_start + 512 + tid]
            + input[block_start + 768 + tid]
```

The block then runs a sequential-addressing reduction within shared memory. This
cuts the number of global writes and kernel-launch reduction levels while keeping
the within-block reduction on faster shared memory.

The optimized coarsened version keeps the same `K = 4` accumulation and shared
memory layout, but unrolls the fixed global-memory accumulation loop and stops
the shared-memory tree once one warp remains. The final 32 values are reduced
with explicit `__shfl_down_sync` steps, avoiding the last few shared-memory
accesses and block-wide synchronizations.

`thrust_reference_sum` uses `thrust::reduce` as a library baseline. It is not the
mechanism being studied here; it is useful for checking correctness and seeing
how far the educational kernels are from a tuned library implementation.

## Caveats

Floating-point addition is not associative, so the GPU tree order can differ
slightly from a CPU left-to-right sum. `__syncthreads()` only synchronizes the
threads in a block, so each block writes one partial sum and the host wrapper
launches the same kernel repeatedly until one value remains.

The host convenience wrappers allocate and copy for tests. The benchmark uses the
device-pointer APIs and reusable `DeviceReductionBuffers` to keep allocation and
host-to-device input copies out of the timed loop.
