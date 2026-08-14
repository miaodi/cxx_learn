# Expression Templates and Automatic Differentiation

## Concept

This course builds a mathematical expression as a C++ type. For example,

```cpp
auto equation = x * x + 2.0;
```

creates a value whose type is structurally equivalent to:

```cpp
Add<Multiply<Variable<0>, Variable<0>>, Constant<double>>
```

The type records the operations; the object stores runtime data such as the
value `2.0`. This gives template argument deduction, concepts, value semantics,
partial specialization, recursive type transformations, and `constexpr` work a
single concrete purpose.

The learner headers are intentionally incomplete. Tests act as compile-time
specifications, while a complete reference implementation lives outside the
default include path under `solutions/`.

## Learning Loop

For each milestone:

1. Configure CMake with that milestone enabled.
2. Read its test before editing the corresponding header.
3. Predict the expression type and evaluation result.
4. Implement only the next missing behavior.
5. Read the first useful compiler error and identify the failed substitution or
   constraint.
6. Build `expr_ad_all`, then run the focused CTest tests.
7. Answer the checkpoint questions before advancing.

Increasing `EXPR_AD_MILESTONE` enables that milestone and all earlier ones. A
newly enabled milestone is expected to fail until its scaffold is completed.

## What To Run

Start from the `cxx_learn` root. Milestone 0 verifies that the empty umbrella
header is structurally valid:

```sh
cmake -S . -B build-expr-ad -DEXPR_AD_MILESTONE=0
cmake --build build-expr-ad --target expr_ad_all
ctest --test-dir build-expr-ad -R '^expr_ad_' --output-on-failure
```

Select milestone 1 when you are ready to implement the first nodes:

```sh
cmake -S . -B build-expr-ad -DEXPR_AD_MILESTONE=1
cmake --build build-expr-ad --target expr_ad_all
```

After completing milestone 6:

```sh
cmake -S . -B build-expr-ad -DEXPR_AD_MILESTONE=6
cmake --build build-expr-ad --target expr_ad_all
ctest --test-dir build-expr-ad -R '^expr_ad_' --output-on-failure
```

To verify or inspect the reference implementation in a separate build tree:

```sh
cmake -S . -B build-expr-ad-reference \
  -DEXPR_AD_MILESTONE=6 \
  -DEXPR_AD_USE_REFERENCE=ON
cmake --build build-expr-ad-reference --target expr_ad_all
ctest --test-dir build-expr-ad-reference -R '^expr_ad_' --output-on-failure
```

## Milestone 1: Types Are the Tree

Implement `include/expr_ad/core.hpp` against `tests/test_nodes.cpp`:

- `is_expression`, `is_expression_v`, and `Expression`
- `Constant<T>` and `Variable<Index>`
- `Zero` and `One`
- `Add<L, R>` and `Multiply<L, R>`
- free `evaluate(expression, values...)`

Store children directly by value. `Variable<Index>` selects its value from the
argument pack, while constants ignore that pack.

Hints:

- Give every node an `expression_tag` member type.
- Detect that member first with `std::void_t`, then define the concept from the
  resulting trait. This makes the older SFINAE mechanism and the C++20 surface
  syntax directly comparable.
- `std::tie(values...)` and `std::get<Index>` make the variable lookup visible.

Checkpoint questions:

- Which parts of an equation are encoded in the expression's type?
- Which parts must remain runtime object state?
- Why does storing children by value avoid dangling references?
- At what point is an invalid variable index rejected?

## Milestone 2: Constrained Expression Builders

Implement `include/expr_ad/operators.hpp` against
`tests/test_operators.cpp`:

- `constant(value)` plus predefined variables `x` and `y`
- `ArithmeticValue` and `ExpressionOperand`
- conversion of either an expression or scalar into an owned expression
- constrained `operator+` and `operator*`

At least one operand must already be an expression. Consequently, `x + 2`
builds a tree while ordinary `1 + 2` continues to use the built-in operator.

Checkpoint questions:

- Why must stored operand types remove references and cv-qualification?
- What overloads would this namespace accidentally capture without the
  "one expression operand" constraint?
- How does the `Expression` concept improve the diagnostic compared with raw
  `std::enable_if_t`?

## Milestone 3: Unary and Nonlinear Operations

Implement `include/expr_ad/functions.hpp` against
`tests/test_functions.cpp`:

- `Subtract`, `Divide`, and `Negate`
- `Sine`, `Cosine`, and `Exponential`
- constrained builders for binary `-`, `/`, unary `-`, `sin`, `cos`, and `exp`

Use unqualified math calls after `using std::sin`, `using std::cos`, or
`using std::exp`. This supports ordinary scalars now and enables argument-
dependent lookup to find dual-number overloads later.

Checkpoint questions:

- Why is `Sine<E>` a different type from `Cosine<E>` even if their storage is
  identical?
- Why does an unqualified dependent call matter for later extensibility?
- Which operations can participate in constant evaluation under the selected
  compiler and C++20 standard-library implementation?

## Milestone 4: Symbolic Differentiation

Implement `include/expr_ad/differentiate.hpp` against
`tests/test_differentiate.cpp`:

- primary template `derivative<Expression, Variable>`
- convenience alias `derivative_t`
- `differentiate<Index>(expression)`
- partial specializations for every leaf, binary node, and unary function

Each specialization computes an output `type` and provides `make()` to
construct that output from the runtime values in the original tree. Implement
the sum, product, quotient, negation, sine, cosine, and exponential rules.

Checkpoint questions:

- Why is partial specialization a form of pattern matching on syntax?
- Why must the product rule retain copies of both original operands?
- What information would a type alias alone lose when constants contain
  runtime values?
- Why does the unspecialized primary template give a useful extension error?

## Milestone 5: Local Type Rewriting

Implement `include/expr_ad/simplify.hpp` against
`tests/test_simplify.cpp`:

- `simplifier<Expression>`, `simplified_t`, and `simplify(expression)`
- recursive child simplification
- `0 + e -> e`, `e + 0 -> e`
- `1 * e -> e`, `e * 1 -> e`, and `0 * e -> 0`
- `e - 0 -> e`, `e / 1 -> e`, and `-(-e) -> e`

Keep the rules local and deterministic. This milestone is not a general
computer-algebra system and should not reorder terms or combine like terms.

Checkpoint questions:

- Why must children be simplified before selecting the parent rule?
- Why can a runtime-valued `Constant<T>` not drive type-level zero elimination?
- Which rewrites change floating-point behavior for NaN, infinity, signed zero,
  or exceptions?

## Milestone 6: Forward-Mode AD

Implement `include/expr_ad/dual.hpp` against `tests/test_forward_ad.cpp`:

- `Dual<T>{value, derivative}`
- addition, subtraction, multiplication, division, and unary negation
- dual overloads of `sin`, `cos`, and `exp`

Seed one independent variable with derivative `1` and all others with `0`.
Evaluate the unchanged expression tree using dual values, then compare the
result with its symbolic derivative and a central finite difference.

Checkpoint questions:

- Why does one dual evaluation compute one directional derivative?
- How does operator overloading propagate the chain rule without transforming
  the expression type?
- When would symbolic differentiation cause a much larger type than forward
  mode?
- Why is finite difference only a numerical check rather than an exact oracle?

## Diagnostic Exercises

After finishing milestone 3, compile these files directly against the reference
headers and inspect the first user-code diagnostic:

```sh
c++ -std=c++20 -IExpressionTemplatesAD/solutions/include \
  -fsyntax-only ExpressionTemplatesAD/negative/unsupported_operand.cpp

c++ -std=c++20 -IExpressionTemplatesAD/solutions/include \
  -fsyntax-only ExpressionTemplatesAD/negative/missing_binding.cpp
```

Both commands are expected to fail. The first violates the arithmetic/expression
operand constraint; the second supplies too few values for `Variable<1>`.

## What To Look For

- `decltype(equation)` mirrors the equation's syntax without heap allocation or
  virtual dispatch.
- Symbolic differentiation creates a new, initially verbose expression type.
- Simplification visibly reduces that type through recursive compile-time rules.
- The same evaluator accepts `double` and `Dual<double>` because its operations
  are generic.
- Symbolic, forward-mode, and finite-difference derivatives agree within the
  documented floating-point tolerance at nonsingular test points.

## Caveats

- Expression types grow with equation size, increasing diagnostic length,
  compile time, and potentially generated code size.
- Value ownership is deliberately safe and simple, but building a new node can
  copy an existing subtree. Production expression-template libraries often use
  more elaborate closure policies.
- The identities involving zero assume ordinary finite algebra. In IEEE 754,
  replacing `0 * infinity` or `0 * NaN` with `0` changes behavior.
- Division has no domain tracking; callers must avoid zero denominators.
- Central finite differences balance truncation and rounding error and should
  not be expected to match exact derivatives bit-for-bit.
- Reverse-mode AD needs a runtime tape or computation graph and is intentionally
  a separate future project.

## Extensions

- Add a `Power<Base, IntegerExponent>` node and derive its specialization.
- Inspect optimized assembly for a hand-written equation and its expression-tree
  equivalent.
- Build reverse mode separately and compare its cost for many-input,
  single-output functions.
