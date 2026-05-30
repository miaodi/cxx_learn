# C++ Learning Project Context

This repository is a learning workspace for experienced programmers studying
systems, performance, and architecture concepts through small C++ and CUDA
examples. It is not primarily a production library. Prefer demonstrative,
readable examples that expose the important mechanism over broad abstractions or
maximum generality.

Current areas include CPU behavior, operating-system-visible effects, computer
architecture, hardware performance, GPU programming, parallelism, memory models,
and modern C++ language techniques.

## Repository Map

- `Cache/`: CPU cache hierarchy, locality, prefetching, stride effects, and
  memory access latency demonstrations.
- `RegisterLatency/`: instruction latency and register-level performance
  experiments.
- `Branch/`: branch prediction and control-flow performance examples.
- `FalseSharing/`: cache coherence and shared-memory parallelism pitfalls.
- `Atomic/`: atomics, memory ordering, and data-race demonstrations.
- `ParallelSort/`: parallel sorting and radix-sort experiments.
- `GEMM/`: CPU matrix multiplication kernels progressing from naive loops to
  blocking, packing, register blocking, and SIMD variants.
- `MatrixTranspose/`: memory-layout and cache behavior in transpose kernels.
- `CrossProductVecLen/`: vector operations, FMA, and SIMD-oriented benchmarks.
- `CopyVsMemcpy/`, `SkipCopy/`, `BitProxy/`, `Pow/`, `Search/`: focused
  microbenchmarks for common C++ and performance questions.
- `AssemblyStudy/`: small examples for inspecting generated assembly.
- `TemplateMetaprogramming/`, `TypeDeduction/`: modern C++ type-system and
  compile-time programming examples.
- `TwoComplement/`: integer representation and low-level arithmetic examples.
- `cuda/`: CUDA examples covering GPU information, memory allocation, memory
  stride, bank conflicts, permutation, graph transpose, Gram-Schmidt, sorting,
  and PMPP-style kernels.
- `cuda/PMPP/`: CUDA programming exercises such as vector add, convolution,
  reduction, merge, radix sort, and GPU GEMM.
- `misc/`: miscellaneous small experiments and utility examples.

## Build And Test

The project uses CMake with C++20.

Common CPU-only configure and build:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

Tests and benchmarks are built only when their dependencies are available.
Missing GoogleTest or google/benchmark targets are skipped by default. To let
CMake download missing test and benchmark dependencies:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCXX_LEARNING_FETCH_DEPS=ON
```

CUDA is optional and disabled by default:

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=ON
cmake --build build-cuda -j
```

Run available tests with:

```sh
ctest --test-dir build --output-on-failure
```

## Development Guidance

- Keep examples small, direct, and easy to inspect. The point is to teach the
  underlying concept, not to hide it behind a reusable framework.
- Prefer one concept per example or benchmark. If a change demonstrates several
  ideas, split the code or explain the interaction explicitly.
- For each new example directory, add a short `README.md` when practical. The
  explanation should cover the concept, what to run, what result pattern to
  expect, and why the result demonstrates the mechanism.
- For existing example directories, update the local `README.md` when behavior,
  assumptions, run commands, or interpretation changes.
- Favor simple, reproducible inputs and benchmark cases before adding elaborate
  tuning knobs.
- Treat benchmark numbers as hardware-, compiler-, and build-dependent. Document
  expected trends and performance cliffs rather than promising exact timings.
- Preserve correctness checks for optimized examples. When adding a faster
  implementation, compare it against a simple baseline where possible.
- Use clear names that expose the teaching progression, such as `naive`,
  `blocked`, `packed`, `tiled`, `prefetch`, or `memory_order_relaxed`.
- Keep comments focused on non-obvious architecture, compiler, memory-model, or
  GPU behavior. Avoid comments that restate ordinary C++ syntax.
- Avoid broad refactors unless they make the learning progression clearer.
- When adding new examples, directories, dependencies, targets, or learning
  areas, keep this `AGENTS.md` file consistent with the new project structure
  and expectations.

## C++ Guidance

- Use C++20 and preserve the existing CMake style.
- Keep benchmark kernels explicit enough that a professional programmer can map
  source code to generated assembly, cache behavior, or GPU execution behavior.
- For CPU performance examples, reason about cache lines, prefetching, branch
  prediction, vectorization, false sharing, alignment, allocation, and
  measurement overhead.
- For atomics and concurrency examples, make the happens-before relationship,
  memory order, race condition, and intended observation explicit.
- For template and type-system examples, prefer compact demonstrations with
  visible compile-time behavior over large generic libraries.
- When changing performance-sensitive code, avoid accidental extra allocation,
  synchronization, copies, virtual dispatch, or hidden benchmark setup work
  inside the measured region.

## CUDA Guidance

- Keep CUDA examples explicit about grid shape, block shape, memory access
  pattern, shared-memory usage, synchronization, and host-device transfer scope.
- Separate correctness tests from benchmark timing when possible.
- For GPU performance examples, reason about coalescing, occupancy, shared
  memory, bank conflicts, register pressure, warp divergence, launch overhead,
  and transfer overhead.
- Keep `USE_CUDA=OFF` builds working unless the task is explicitly CUDA-only.
- When adding CUDA examples under `cuda/PMPP/`, follow the existing style of
  pairing simple kernels with tests, benchmarks, and explanations.

## Documentation Expectations

When adding or changing an example, prefer this explanation shape:

```text
# Example Name

## Concept
What system, architecture, language, or hardware concept this demonstrates.

## What To Run
The target, executable, benchmark, or test command.

## What To Look For
The expected trend, output shape, or comparison.

## Why It Happens
The CPU, OS, compiler, C++, CUDA, or hardware mechanism behind the result.

## Caveats
Hardware, compiler, optimization level, dependency, or measurement limitations.
```

The explanation does not need to be long. A precise short note is better than a
large tutorial that obscures the example.

## Verification Guidance

- For code changes, build the affected target when possible.
- For tests, run the focused test or `ctest --test-dir build --output-on-failure`
  if the build tree already exists.
- For benchmarks, verify that the benchmark builds and runs a small case before
  trusting numbers.
- For CUDA changes, verify both the relevant CUDA target and that CPU-only
  configuration still works when the change touches shared CMake files.
- If dependencies or hardware are missing, state what was not run and why.

## AI Assistant Guidance

When working in this repository, assume the user values learning and clear
technical explanation. Before editing, identify the concept the change is meant
to teach. After editing, summarize both the code change and the concept it
demonstrates.

Good requests for this repo look like:

```text
Add a small example showing why false sharing hurts OpenMP scaling, with a
README that explains the cache-line effect.
```

```text
Create a CUDA bank-conflict benchmark and explain the expected shared-memory
access pattern.
```

```text
Add a CPU GEMM variant between blocked and packed kernels that demonstrates why
packing B improves locality.
```

## AGENTS.md Maintenance Policy

Update this file when the repository structure, build commands, major learning
areas, dependency behavior, or durable development expectations change. Keep it
focused on guidance future agents should follow, not temporary task notes.
Before finishing a change, check whether the change makes any section of this
file stale, incomplete, or misleading, and update it in the same change when it
does.
