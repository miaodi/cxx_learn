# Atomic Examples

Small C++20 examples for atomics, memory ordering, data races, and
happens-before relationships.

## Build

Configure once from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Build and run one standalone example:

```sh
cmake --build build --target <target>
./build/Atomic/<target>
```

## Learning Map

| Topic | Executable | Notes |
| --- | --- | --- |
| Acquire/release message passing and relaxed store buffering | `memory_order_demo` | [`notes/memory_order.md`](notes/memory_order.md) |

## Existing Tests

The `Atomic` test target contains older GTest-based experiments in `test.cpp`,
`memory_order_test.cpp`, and `benign_data_racing_test.cpp`. It is built only
when the project finds both GTest and google/benchmark.

## Checklist

After these examples, you should be able to explain:

- why default atomic operations are `memory_order_seq_cst`;
- why `memory_order_relaxed` gives atomicity but not cross-location ordering;
- how acquire/release publishes ordinary data through an atomic flag;
- why a relaxed litmus test can produce results forbidden by `seq_cst`;
- why a non-atomic payload plus a relaxed flag is a data race, not just a weakly ordered program.

## Caveats

Memory-order observations are compiler-, CPU-, optimization-, and timing-
dependent. A result that is allowed by the C++ memory model may still be rare or
unobserved on a particular machine.
