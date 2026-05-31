# Float32 Bit Layout

## Concept

This example introduces the IEEE-754 float32 bit layout in small sections. The
first section focuses on the sign bit and signed zero. The second section shows
the special encodings for infinity and NaN. The third section compares normal
and subnormal finite values.

A float32 value has 32 bits split into three fields:

```text
sign | exponent | fraction
  1  |    8     |    23
```

Unlike two's-complement integers, IEEE-754 floating point has a true sign bit.
That means zero has two encodings: `+0.0f` and `-0.0f`.

## What To Run

```sh
cmake --build build --target ieee_float32
./build/IEEEFloat/ieee_float32
```

## What To Look For

Section 1 prints `+0.0f` and `-0.0f`. The two zeros have identical exponent and
fraction fields. Only the sign bit is different:

```text
+0.0f  : 0 | 00000000 | 00000000000000000000000
-0.0f  : 1 | 00000000 | 00000000000000000000000
```

The values still compare equal:

```text
+0.0f == -0.0f: true
```

But the sign is observable through `std::signbit` and through operations such as
division by zero:

```text
1.0f / +0.0f: inf
1.0f / -0.0f: -inf
```

Section 2 prints infinity and NaN. These are special encodings where the exponent
field is all ones:

```text
+inf   : 0 | 11111111 | 00000000000000000000000
-inf   : 1 | 11111111 | 00000000000000000000000
NaN    : 0 | 11111111 | nonzero fraction bits
```

Infinity has an all-ones exponent and an all-zero fraction. NaN has an all-ones
exponent and a nonzero fraction. The exact NaN fraction bits can vary, but they
must be nonzero.

Section 3 prints one ordinary normal number, the smallest positive normal number,
and the smallest positive subnormal number:

```text
1.5f   : 0 | 01111111 | 10000000000000000000000  raw=0x3FC00000
min norm: 0 | 00000001 | 00000000000000000000000  raw=0x00800000
min sub: 0 | 00000000 | 00000000000000000000001  raw=0x00000001
```

The important difference is the significand `M`:

```text
normal:    value = (-1)^sign * 1.fraction * 2^(e - bias)
subnormal: value = (-1)^sign * 0.fraction * 2^(1 - bias)
```

For float32, the bias is `127`. A normal number stores an exponent field `e` from
`1` to `254` and gets an implicit leading `1` in `M`. A subnormal number has
`e = 0`, keeps the same smallest exponent `E = 1 - bias = -126`, and uses no
implicit leading `1`.

The example also compares two bit patterns with the same stored fraction field:

```text
e=1 frac=1: 0 | 00000001 | 00000000000000000000001  raw=0x00800001
e=0 frac=1: 0 | 00000000 | 00000000000000000000001  raw=0x00000001
```

Both have fraction bits equal to integer `1`, but the decoded significand is very
different. The normal value gets the implicit leading bit, so `M = 1 + 2^-23`.
The subnormal value does not, so `M = 2^-23`.

Both examples use the same effective exponent `E = -126`. If you ignore that
common `2^E` factor and compare only the significand part, the gap is exactly
the hidden leading bit:

```text
(1 + 2^-23) - 2^-23 = 1
```

## Why It Happens

IEEE-754 represents a floating-point number with a sign field, an exponent field,
and a fraction field. The sign field is separate from the magnitude fields. For
zero, both the exponent and fraction fields are all zero. The sign bit can still
be either `0` or `1`, producing `+0.0f` and `-0.0f`.

This is different from two's-complement integers. Integer two's complement uses
all bit patterns as part of one modular encoding and has only one zero. Float32
keeps a separate sign bit, so it naturally has both positive and negative zero.

Signed zero is useful because it preserves directional information around zero.
For example, a computation that underflows from a tiny negative value can produce
`-0.0f`, and later operations can still observe that sign.

Infinity and NaN use the exponent field value `255` (`11111111`) instead of the
normal finite-number formula. The fraction field separates the cases: zero
fraction means infinity, and nonzero fraction means NaN. NaN stands for "not a
number" and represents invalid or indeterminate results. A useful check is that
NaN does not compare equal to itself.

Normal and subnormal finite numbers use different rules so float32 can taper
gradually toward zero. The smallest positive normal has `e = 1`, `E = -126`, and
`M = 1.0`, so its value is `1.0 * 2^-126`. The smallest positive subnormal has
`e = 0`, `E = -126`, and `M = 1 / 2^23`, so its value is `2^-149`. Without
subnormals, values between zero and `2^-126` would underflow directly to zero.

## Interview Takeaway

IEEE float is sign/exponent/fraction encoding, not a scaled integer. Special bit
patterns explain signed zero, infinities, NaNs, normal values, and subnormal
gradual underflow.

## Caveats

This example assumes `float` is IEEE-754 binary32 and checks that assumption with
`std::numeric_limits<float>::is_iec559` and `sizeof(float) == 4`. Decimal output
for very small values is formatted by the standard library, so the exact number
of printed digits can vary by implementation.
