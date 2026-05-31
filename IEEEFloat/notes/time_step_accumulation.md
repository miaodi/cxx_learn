# Time-Step Accumulation

## Concept

Fixed-step simulations often update time like this:

```cpp
time += dt;
```

If `time` and `dt` are floating-point values, every step rounds. Over many steps,
those small rounding errors can accumulate into visible drift. The longer the
simulation runs, the more rounded additions are chained together.

## What To Run

```sh
cmake --build build --target ieee_float_time_step_accumulation
./build/IEEEFloat/ieee_float_time_step_accumulation
```

## What To Look For

The example uses a fixed time step of `1/60` second and samples the accumulated
time after 1 second, 1 minute, 10 minutes, 1 hour, 6 hours, and 24 hours.

It compares three methods:

```text
float time += float dt
float time = steps * float dt
double time += double dt
```

The repeated `float` accumulation should drift much more than the `steps * dt`
calculation. The `double` accumulation also rounds, but the error is much smaller
for this scale.

## Why It Happens

`1/60` is not exactly representable as a binary floating-point value. The stored
`dt` is already rounded. Then every `time += dt` operation rounds the partial sum
again. A long run is therefore not one exact multiplication followed by one
rounding; it is thousands or millions of rounded additions.

As `time` grows, the spacing between adjacent representable `float` values near
`time` also grows. The same `dt` is added to a coarser grid, so each addition may
round slightly up or down relative to the ideal time.

Computing `time` from an integer step count:

```cpp
time = step_count * dt;
```

still rounds, but it avoids the long dependency chain of previous rounded time
values. This is often preferable when a simulation already has an integer tick or
frame counter.

## Interview Takeaway

Repeated floating-point updates can accumulate drift. For fixed-step simulations,
store the authoritative step count as an integer when possible, and compute time
from that count instead of treating a floating-point clock as the source of truth.

## Caveats

The exact drift depends on the step size, precision, duration, compiler flags,
and hardware. `double` reduces the error but does not make the process exact.
Compiler flags such as `-ffast-math` may allow transformations that change the
observed behavior.
