# Decimal Rounding

## Concept

Most finite decimal fractions are repeating fractions in base 2. When C++ parses a
literal such as `0.1f` or `0.1`, the value stored in `float` or `double` is the
nearest representable binary floating-point value, not the exact real number
`1/10`.

## What To Run

```sh
cmake --build build --target ieee_float_decimal_rounding
./build/IEEEFloat/ieee_float_decimal_rounding
```

## What To Look For

The program prints `0.1`, `0.2`, `0.3`, and their sums with enough digits to show
the stored values. The `float` case may show `0.1f + 0.2f == 0.3f` as `true`,
while the `double` case shows the classic `0.1 + 0.2 == 0.3` as `false`.

That difference is useful: it shows that floating-point surprises depend on the
format, the exact values, and the final rounding step.

## Why It Happens

Binary floating point represents numbers as a sign, a binary significand, and a
power-of-two exponent. Fractions whose denominator has factors other than 2, such
as `1/10`, repeat forever in binary. The stored value is rounded to the nearest
available bit pattern.

## Interview Takeaway

Floating point is not decimal arithmetic. Decimal-looking literals are often
already approximations before any computation starts.

## Caveats

The printed decimal differences use `long double` as a higher-precision reference
for readability. `long double` is still finite precision and is platform
dependent.
