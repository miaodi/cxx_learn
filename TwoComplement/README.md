# Two's Complement Encoding

## Concept

This example shows both the rule for two's complement encoding and evidence that
`std::int8_t` uses that representation on the current platform. Positive values
use the same bit pattern as ordinary binary. A negative value `x` in `N` bits is
encoded as `2^N + x`.

## What To Run

Build and run the example target:

```sh
cmake --build build --target two_complement_encode
./build/TwoComplement/two_complement_encode
```

If the build directory does not exist yet, configure it first:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

## What To Look For

The first part prints several decimal values encoded as 8-bit two's complement
bit patterns. The key examples are:

```text
  -1 encoded as  8 bits: 0b11111111 = 0xFF
  -2 encoded as  8 bits: 0b11111110 = 0xFE
  -5 encoded as  8 bits: 0b11111011 = 0xFB
-128 encoded as  8 bits: 0b10000000 = 0x80
```

The second part inspects the actual object representation of `std::int8_t` with
`std::bit_cast`. If the printed object bits match the computed two's complement
encoding, this platform stores `std::int8_t` as two's complement.

## Why It Happens

With `N` bits there are `2^N` possible bit patterns. Two's complement assigns
the lower half to non-negative values and the upper half to negative values. In
8 bits, `-5` is encoded as `2^8 - 5 = 251`, which is `11111011` in binary.

Compared with a separate sign-bit representation, two's complement does not
waste one bit pattern on negative zero. A sign-bit scheme has both `+0` and
`-0`, while two's complement has only one zero and uses the extra pattern for
one more negative value. That is why 8-bit two's complement covers `-128` to
`127`, not `-127` to `127`.

This representation makes addition and subtraction use the same binary hardware
for signed and unsigned integers. For example, adding `5` and `-5` gives
`00000101 + 11111011 = 1_00000000`; keeping only the low 8 bits leaves zero.

## Caveats

The standard library has `std::int8_t`, but not `std::int4_t`. The fixed-width
integer typedefs exist only when the implementation provides an exact-width
integer type, and ordinary C++ implementations do not provide addressable 4-bit
integer objects.

The example computes the expected bit patterns explicitly instead of relying on
signed integer overflow behavior. It uses widths up to 31 bits so the
intermediate `1 << (width - 1)` calculation stays within the range of `int`.
