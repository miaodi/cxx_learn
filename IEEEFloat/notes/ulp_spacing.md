# ULP Spacing

## Concept

An ulp is a unit in the last place: the gap between adjacent representable values
near a particular floating-point number. Float spacing is not constant. It grows
as the exponent grows.

## What To Run

```sh
cmake --build build --target ieee_float_ulp_spacing
./build/IEEEFloat/ieee_float_ulp_spacing
```

## What To Look For

The program uses `std::nextafter` to print the next larger float after `1.0f`,
`2.0f`, `1024.0f`, and `16777216.0f`. The printed `delta` is:

```text
next larger float - x
```

For positive normal floats, this delta is nondecreasing as `x` grows. It stays
constant within a binade, the interval between adjacent powers of two, then
roughly doubles when `x` crosses the next power of two.

The second section shows:

```text
16777216.0f + 1.0f == 16777216.0f
16777216.0f + 2.0f != 16777216.0f
```

Near `2^24`, the delta to the next larger float is `2`, so adding `1` is below
the current spacing.

## Why It Happens

Float32 has 24 bits of precision when including the implicit leading bit. Every
integer up to `2^24` is exactly representable, but after that point the format
cannot represent every adjacent integer. The representable grid becomes spaced by
`2`, then `4`, then `8`, and so on as the exponent grows.

## Interview Takeaway

Machine epsilon is local to `1.0`; precision is relative. A float has a fixed
number of significant binary bits, not a fixed number of decimal places.

## Caveats

The exact decimal formatting is implementation-dependent, but the raw bit patterns
and spacing trend should be stable on IEEE-754 binary32 systems.
