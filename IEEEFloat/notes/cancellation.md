# Cancellation

## Concept

Cancellation happens when nearly equal values are subtracted. The leading digits
cancel, so the result is built from the low-order digits of the inputs. If those
inputs were already rounded, the result can have very few trustworthy digits.

## What To Run

```sh
cmake --build build --target ieee_float_cancellation
./build/IEEEFloat/ieee_float_cancellation
```

## What To Look For

The first section subtracts nearby float values. The result is determined by the
stored float bit patterns, not by the decimal text a programmer had in mind.

The second section compares two mathematically equivalent expressions:

```text
sqrt(x + 1) - sqrt(x)
1 / (sqrt(x + 1) + sqrt(x))
```

For large `x`, the direct formula subtracts two nearly equal square roots. The
rationalized formula avoids that subtraction and stays close to the higher-
precision reference.

## Why It Happens

Floating-point values are rounded before subtraction happens. If the meaningful
answer is small compared with the inputs, most of the shared leading bits cancel
out. Any earlier rounding error can become a large fraction of the final result.

The rationalized square-root expression moves the subtraction out of the formula:

```text
sqrt(x + 1) - sqrt(x)
= ((x + 1) - x) / (sqrt(x + 1) + sqrt(x))
= 1 / (sqrt(x + 1) + sqrt(x))
```

This computes the same real-number expression but avoids subtracting nearly equal
rounded square roots.

## Interview Takeaway

The problem is not subtraction itself. The problem is subtracting nearly equal
rounded values when the small difference is the result you care about. Stable
reformulations can be more important than using a wider type.

## Caveats

The stable formula is still floating-point arithmetic and still rounds. It is not
exact; it is just much better conditioned for this input. Compiler flags that
reassociate expressions may change the demonstration.
