# IEEE Floating Point Examples

Small C++20 examples for IEEE-754 representation, rounding behavior, numerical
stability, and interview-focused floating-point questions.

## Build

Configure once from the repository root:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

Build and run one example:

```sh
cmake --build build --target <target>
./build/IEEEFloat/<target>
```

## Interview Map

| Topic | Executable | Notes |
| --- | --- | --- |
| Float32 layout | `ieee_float32` | [`notes/float32_layout.md`](notes/float32_layout.md) |
| Kahan summation and algebra | `ieee_float_kahan_sum` | [`notes/kahan_sum.md`](notes/kahan_sum.md) |
| Decimal rounding | `ieee_float_decimal_rounding` | [`notes/decimal_rounding.md`](notes/decimal_rounding.md) |
| ULP spacing | `ieee_float_ulp_spacing` | [`notes/ulp_spacing.md`](notes/ulp_spacing.md) |
| Cancellation | `ieee_float_cancellation` | [`notes/cancellation.md`](notes/cancellation.md) |

## Checklist

After these examples, you should be able to explain:

- why decimal fractions such as `0.1` are usually approximations in binary;
- why float spacing grows as the exponent grows;
- why `large + small` may equal `large`;
- why subtraction of nearly equal rounded values can lose useful digits;
- why addition and multiplication rewrites can change final bits;
- why reduction order changes results;
- what Kahan summation compensates for;
- why `x / a` can differ from `x * (1.0f / a)`;
- why `-ffast-math` can invalidate examples that rely on strict IEEE behavior.

## Caveats

These examples assume ordinary IEEE-754 behavior and default round-to-nearest
semantics. Compiler flags such as `-ffast-math`, `-fassociative-math`, or
aggressive reciprocal/FMA transformations can change the observations.
