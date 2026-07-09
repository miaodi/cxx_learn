# PMPP Scan

## Concept

This folder is the PMPP Chapter 11 learning scaffold for parallel prefix sum,
also called scan. A scan converts an input array into all prefix sums:

```text
input:          3   1   7   0   4   1
inclusive scan: 3   4  11  11  15  16
exclusive scan: 0   3   4  11  11  15
```

The initial API focuses on inclusive `float` scans over device pointers.
`brent_kung_inclusive_scan` now implements the single-block case. The other
functions are placeholders: they validate the call shape, accept empty inputs,
and return `cudaErrorNotSupported` for non-empty scans until their kernels are
implemented.

## Planned Progression

- `kogge_stone_inclusive_scan`: many active threads and `log2(n)` dependency
  steps; simple but does more total work.
- `brent_kung_inclusive_scan`: an up-sweep/down-sweep tree with less work but
  more structured synchronization. The current implementation supports one
  `512`-element tile: `256` threads load two elements each.
- `coarsened_brent_kung_inclusive_scan`: each thread scans several elements in
  registers before participating in the block-level tree.
- `hierarchical_inclusive_scan`: scan block tiles, scan block sums, then add the
  scanned block offsets back to each tile.
- `cub_inclusive_scan_reference`: a library reference for correctness and
  performance comparison.

## What To Run

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=RelWithDebInfo -DUSE_CUDA=ON
cmake --build build-cuda --target PMPP_scan -j
cmake --build build-cuda --target PMPP_scan_test -j
ctest --test-dir build-cuda -R InclusiveScan --output-on-failure
```

`PMPP_scan_test` is built only when GoogleTest is available. If not, building
`PMPP_scan` still checks that the scaffold compiles.

## What To Look For

At this stage, the Brent-Kung tests compare against a CPU inclusive scan for a
small non-power-of-two input and a full `512`-element tile. Empty inputs return
`cudaSuccess`, invalid arguments return `cudaErrorInvalidValue`, and Brent-Kung
inputs larger than one tile return `cudaErrorNotSupported`.

As each remaining algorithm is implemented, replace its placeholder assertions
with CPU reference comparisons on small arrays, non-power-of-two sizes, and
larger arrays that require multiple blocks.

## Why It Happens

Scan is harder than reduction because every output element needs a different
partial result, not just one final aggregate. Kogge-Stone exposes the dependency
distance doubling pattern directly. Brent-Kung reduces unnecessary arithmetic by
building partial sums in an up-sweep, then distributing those partial sums in a
down-sweep. The current Brent-Kung kernel pads missing elements in the tile with
zero, so non-power-of-two sizes below `512` still produce the correct prefixes.

Hierarchical scan is the multi-block version: each block can scan locally, but
block offsets must be computed and added before the global result is correct.

## Caveats

Floating-point scan results depend on the tree order, so GPU scans may differ
slightly from a CPU left-to-right reference. The future benchmark should keep
allocation and host-device copies outside the timed region, because scan kernels
are memory-bandwidth sensitive.
