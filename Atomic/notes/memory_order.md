# Memory Order Basics

## Concept

This example demonstrates two separate ideas in the C++ memory model:

- `memory_order_release` and `memory_order_acquire` can publish ordinary data from one thread to another.
- `memory_order_relaxed` gives atomicity for the atomic object, but does not create ordering between different memory locations.

The code is in `../memory_order_demo.cpp`.

## What To Run

Configure and build from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target memory_order_demo
```

Run the demo:

```sh
./build/Atomic/memory_order_demo
```

You can pass a larger iteration count for the relaxed litmus test:

```sh
./build/Atomic/memory_order_demo 1000000
```

## What To Look For

The first part is deterministic:

```text
acquire/release message passing: observed payload = 42 (guaranteed)
```

The producer writes an ordinary `int payload`, then publishes `ready` with a release-store. The consumer waits with an acquire-load. Once the consumer sees `ready == true`, the write to `payload` is guaranteed to be visible.

The second part is a store-buffering litmus test:

```cpp
// Thread 1
x.store(1, order);
r1 = y.load(order);

// Thread 2
y.store(1, order);
r2 = x.load(order);
```

With `memory_order_relaxed`, the output may include this result:

```text
r1 = 0, r2 = 0
```

That means each thread missed the other thread's store. This result is allowed with relaxed ordering.

With `memory_order_seq_cst`, `r1 = 0, r2 = 0` should stay at zero occurrences. Sequential consistency forces all participating atomic operations into one global order, and that global order cannot explain both loads seeing the initial value.

## Why It Happens

`memory_order_release` and `memory_order_acquire` form a synchronizes-with relationship when the acquire-load reads the value written by the release-store. In the message-passing example, that creates a happens-before relationship:

```text
payload = 42
ready.store(true, release)
ready.load(acquire) sees true
read payload
```

`memory_order_relaxed` does not create that relationship. It only says that each atomic operation is indivisible and participates in the modification order of its own atomic object.

The store-buffering test uses only atomics, so it has no data race. The surprising relaxed outcome is not undefined behavior. It is a legal result of using atomics without asking C++ to order operations on `x` relative to operations on `y`.

## Caveats

The relaxed `r1 = 0, r2 = 0` result is allowed by C++, but it is hardware- and timing-dependent. It may appear often on some CPUs, rarely on others, or not at all in a short run.

Do not demonstrate relaxed message passing by using a non-atomic payload with a relaxed flag. That version has an unsynchronized non-atomic read/write pair, which is a data race and therefore undefined behavior.

`memory_order_seq_cst` is stronger than necessary for many real producer/consumer cases. The message-passing part uses acquire/release because that is the minimal ordering needed to publish data through a flag.

## Extensions

Try these follow-up experiments:

- Replace the acquire/release operations in the message-passing example with default atomic operations and observe that the program is still correct because the default is `memory_order_seq_cst`.
- Increase the litmus-test iteration count and compare results on x86, ARM, and different optimization levels.
