# Concurrency

Small C++20 experiments exploring shared state, synchronization, and the C++
memory model.

## Learning Map

| Topic | Guiding question |
| --- | --- |
| [Atomic memory model](AtomicMemoryModel/README.md) | What ordering and visibility guarantees exist between threads? |

The planned `AtomicVsLock/` guided lab will explore atomic operations versus
critical sections, compound invariants, spinlocks, and contention costs.

## Build

From the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target memory_order_demo
./build/Concurrency/AtomicMemoryModel/memory_order_demo
```

See each topic's README for its experiments, dependencies, and interpretation.
