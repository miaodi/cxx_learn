# Cache Performance Benchmarks

This directory contains simple benchmarks to demonstrate the performance characteristics of different levels of the CPU cache hierarchy.

## What Each Benchmark Shows

### 1. `BM_MemoryLatency` - **The Gold Standard** 🏆
- **Purpose**: Measures pure memory access latency using pointer chasing
- **What to expect**: 
  - ~1-4 ns for L1 cache (4KB-16KB)
  - ~3-12 ns for L2 cache (128KB)  
  - ~12-40 ns for L3 cache (1MB)
  - ~50-300 ns for main RAM (16MB+)
- **Why it works**: Pointer chasing prevents prefetching and measures actual latency

### 2. `BM_SequentialAccess` vs `BM_RandomAccess`
- **Purpose**: Shows the benefit of cache line prefetching
- **What to expect**: Sequential access should be much faster due to automatic prefetching
- **Key insight**: Modern CPUs prefetch entire cache lines (64 bytes), making sequential access very efficient

### 3. `BM_StrideAccess` (different strides)
- **Purpose**: Demonstrates cache line size effects
- **What to expect**: 
  - Stride 1: Fastest (sequential)
  - Stride 16: Still fast (one access per cache line)  
  - Stride 32+: Slower (wasting cache lines)
- **Key insight**: Cache lines are 64 bytes (16 integers), so stride 16 is optimal for sampling

### 4. `BM_MatrixRowMajor` vs `BM_MatrixColumnMajor`
- **Purpose**: Shows cache-friendly vs cache-unfriendly access patterns
- **What to expect**: Row-major should be much faster than column-major
- **Key insight**: C++ stores matrices in row-major order, so accessing by columns causes cache misses

### 5. `BM_NoPrefetch` vs `BM_WithPrefetch`
- **Purpose**: Shows the benefit of manual prefetching for irregular access patterns
- **What to expect**: Prefetch version should be faster for large datasets
- **Key insight**: Software prefetching can help when hardware prefetching fails

## How to Run

```bash
cd build
make cache_bench
./Cache/cache_bench
```

## Interpreting Results

Look for these patterns in your results:
1. **Performance cliffs**: Dramatic slowdowns at cache boundaries (32KB, 256KB, 8MB)
2. **Access pattern effects**: Sequential >> Random access
3. **Cache line effects**: Performance drops when stride exceeds cache line size
4. **Memory layout effects**: Row-major >> Column-major for matrices

## Typical Results (will vary by CPU)

```
BM_MemoryLatency/4KB           2.5 ns
BM_MemoryLatency/16KB          3.1 ns  
BM_MemoryLatency/128KB         8.7 ns    ← L2 boundary
BM_MemoryLatency/1MB          15.2 ns    ← L3 boundary  
BM_MemoryLatency/16MB         89.4 ns    ← RAM access
BM_MemoryLatency/64MB        156.8 ns
```

The key is to see the **performance jumps** at cache boundaries - this clearly demonstrates the cache hierarchy!