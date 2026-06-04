# PMPP BFS

## Concept

This folder is a skeleton for the PMPP Ch. 15.3 vertex-centric top-down BFS.
It intentionally does not maintain a compact frontier array. Each BFS level
launches enough CUDA threads for all vertices, and the implicit frontier is:

```cpp
levels[vertex] == current_level
```

Only those vertices scan their adjacency lists and mark unvisited neighbors for
the next level.

## What To Run

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=ON
cmake --build build-cuda --target PMPP_bfs_bench -j
./build-cuda/cuda/PMPP/bfs/PMPP_bfs_bench
```

To benchmark a different Matrix Market graph:

```sh
PMPP_BFS_MTX=/path/to/graph.mtx ./build-cuda/cuda/PMPP/bfs/PMPP_bfs_bench
```

## What To Look For

The benchmark reads a Matrix Market coordinate file once, builds host CSR with
`cuda/mtx_reader.h`, copies the CSR arrays to the GPU once, and times the BFS
wrapper call plus synchronization. The wrapper initializes `levels` every
iteration, so each benchmark repetition measures a complete BFS from a clean
source state.

## Why It Happens

The simple PMPP version exposes the cost of scanning all vertices at every BFS
level. This is easy to understand and maps directly to a flat CUDA launch, but
it wastes work when the current level contains only a small fraction of the
graph. A later frontier-based version can avoid that by compacting active
vertices between levels.

## Caveats

The sample `.mtx` file is tiny and useful only for plumbing. Use a larger graph
reader into both directions, while `general` input is treated as directed.
