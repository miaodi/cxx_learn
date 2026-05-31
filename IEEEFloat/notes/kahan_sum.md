# Kahan Summation and Floating-Point Algebra

## Concept

`kahan_sum.cpp` demonstrates why ordinary reduction order matters and how Kahan
summation can stabilize a long sum. The example starts with `1.0` and then adds
many copies of `2^-54`. Each tiny term is exactly representable as a `double`,
but it is less than half an ulp near `1.0`, so a naive left-to-right sum rounds
each tiny addition away.

Kahan summation keeps a compensation term for the low-order part that was lost in
the previous rounded addition. That does not make floating-point arithmetic
exact, but it can recover information that a plain running sum repeatedly drops.

The same executable also prints three real-number algebra rules that are not safe
rewrites for strict IEEE floating point:

```text
x * (y - z) != x * y - x * z
(x + y) + z != x + (y + z)
x / a != x * (1.0f / a)
```

The third rule is legitimate: division performs one rounded operation, while
reciprocal-then-multiply rounds the reciprocal and then rounds the multiplication.

## What To Run

```sh
cmake --build build --target ieee_float_kahan_sum
./build/IEEEFloat/ieee_float_kahan_sum
```

## What To Look For

The reduction section compares several ways to add the same values:

```text
naive forward
naive reverse
pairwise
Kahan
long double ref
```

The naive forward sum should stay at `1.0` because every `2^-54` term is rounded
away when it is added directly to a value near `1.0`. Reverse order, pairwise
summation, and Kahan summation should be much closer to the exact result because
they let the tiny terms accumulate before they are lost.

The algebra section prints the input values, both sides of each expression, and
the raw float32 bit patterns. The values differ because each expression rounds at
different points.

## Why It Happens

Floating-point addition rounds after each operation. A long reduction is therefore
not one exact mathematical sum followed by one rounding step; it is a sequence of
rounded partial sums. If a small term is far below the spacing between adjacent
representable values near the current accumulator, adding it may not change the
accumulator at all.

Kahan summation estimates the part that was lost during the previous addition and
subtracts that compensation from the next input term. This lets several small
lost pieces build up until they are large enough to affect the accumulator.

Pairwise summation helps for a different reason. It changes the reduction tree so
that similarly sized terms are often added together before they meet a much
larger partial sum. This is also why parallel reductions can produce different
answers from serial reductions: they use different reduction trees.

The algebra counterexamples are the same issue in smaller form. Real arithmetic
allows reassociation, distribution, and reciprocal replacement. Floating-point
arithmetic rounds intermediate results, so those rewrites can change the final
bits.

## Interview Takeaway

Floating-point reduction is order-sensitive. Kahan summation can reduce rounding
error, but strict algebraic rewrites from real arithmetic are not automatically
valid for floating point.

## Caveats

Kahan summation is not exact arithmetic. It reduces accumulated rounding error,
but it cannot recover information that has already been destroyed by severe
cancellation or by earlier rounded operations.

`long double` is only a higher-precision reference here. It is useful for this
small demonstration, but it is not a proof of exactness in general.

Compiler flags such as `-ffast-math` and `-fassociative-math` allow reassociation
and reciprocal transformations that are invalid under strict IEEE reasoning.
Those flags can change or erase the differences this example is trying to show,
and they can break compensated summation algorithms.
