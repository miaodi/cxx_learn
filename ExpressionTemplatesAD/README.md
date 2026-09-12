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

The expression vocabulary is organized by arity. `core.hpp` contains concepts,
terminal nodes, and the predefined variables; `binary.hpp` contains binary
arithmetic and its builders; and `unary.hpp` contains unary arithmetic and
mathematical functions.

## Progress Checkpoint — Paused / In Progress

The learning work is paused here, not finished. The learner implementation now
includes expression foundations, binary/unary nodes (including `log`), symbolic
differentiation, and initial simplification rules.

At this checkpoint, the existing scaffold, nodes, binary, unary, playground,
differentiation, and simplification tests compile and run with GCC 15.2 in C++20
mode (direct compiler invocations, not a CMake/CTest run). Passing these tests
does not mean every milestone requirement is complete.

When returning:

- Finish milestone 4 coverage: add derivative rules for `Zero` and `One`, needed
  when differentiating trees that already contain symbolic constants.
- Finish milestone 5 recursive simplification for ordinary `Negate`, `Sine`,
  `Cosine`, and `Exponential` nodes; revisit nested unary rewrites.
- Implement milestone 6 in `include/expr_ad/dual.hpp`, which remains a scaffold,
  then run the forward-AD comparisons.
- Use `tests/test_playground.cpp` to resume experiments and the milestone
  commands below to verify the completed work.

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

## Milestone 1: Expression Foundations

Implement `include/expr_ad/core.hpp` against `tests/test_nodes.cpp`:

- `is_expression`, `is_expression_v`, and `Expression`
- `ArithmeticValue` and `Numeric`
- `Constant<T>` and `Variable<Index>`
- predefined variables `x` and `y`
- `Zero` and `One`
- free `evaluate(expression, values...)`

`Constant<T>` stores its value directly. `Variable<Index>` selects its value
from the argument pack, while constants ignore that pack.

Hints:

- Give every node an `expression_tag` member type.
- Detect that member first with `std::void_t`, then define the concept from the
  resulting trait. This makes the older SFINAE mechanism and the C++20 surface
  syntax directly comparable.
- `std::tie(values...)` and `std::get<Index>` make the variable lookup visible.

### Evaluation contract

Every `evaluate` overload receives variable bindings as `const Values&...`.
Evaluation only observes those bindings, and sibling subtrees may reuse the
same binding, as in `x * x`. Passing read-only references avoids copies at each
tree level without forwarding an rvalue through multiple branches.

`Numeric` is a structural concept requiring a copyable type with the usual
arithmetic operations. Unlike `std::is_arithmetic`, it admits user-defined
numeric types such as the `Dual<T>` introduced in milestone 6. The narrower
`ArithmeticValue` concept identifies built-in scalar operands that milestone 2
may convert into constants.

Terminal and composite nodes return evaluation results by value. In particular,
`Variable<Index>` copies the selected binding, while `Add<L, R>` and
`Multiply<L, R>` return their computed results with `auto`. A variable or
composite node therefore has no single intrinsic `value_type`: its result type
depends on the types supplied to `evaluate`. `Constant<T>` may expose
`value_type = T` because that type belongs to its stored state.

The free function follows the same contract and provides the public entry
point:

```cpp
template <Expression E, Numeric... Values>
constexpr auto evaluate(const E& expression, const Values&... values)
    -> decltype(expression.evaluate(values...)) {
  return expression.evaluate(values...);
}
```

Checkpoint questions:

- What structural marker makes `is_expression` recognize a node?
- Which parts of a constant belong to its type and which remain object state?
- Why does `Variable<Index>::evaluate` return its selected binding by value?
- At what point is an invalid variable index rejected?

## Milestone 2: Binary Operations

Implement `include/expr_ad/binary.hpp` against `tests/test_binary.cpp`:

- `Add<L, R>`, `Subtract<L, R>`, `Multiply<L, R>`, and `Divide<L, R>`
- `constant(value)`
- `ExpressionOperand`, building on `ArithmeticValue` from `core.hpp`
- conversion of either an expression or scalar into an owned expression
- constrained binary `operator+`, `operator-`, `operator*`, and `operator/`

At least one operand must already be an expression. Consequently, `x + 2`
builds a tree while ordinary `1 + 2` continues to use the built-in operator.

Checkpoint questions:

- Which parts of an equation are encoded in the expression's type?
- Why does storing child expressions by value avoid dangling references?
- Why must stored operand types remove references and cv-qualification?
- What overloads would this namespace accidentally capture without the
  "one expression operand" constraint?
- How does the `Expression` concept improve the diagnostic compared with raw
  `std::enable_if_t`?

## Milestone 3: Unary Operations

Implement `include/expr_ad/unary.hpp` against `tests/test_unary.cpp`:

- `Negate<E>`, `Sine<E>`, `Cosine<E>`, `Exponential<E>`, and `Logarithm<E>`
- constrained builders for unary `-`, `sin`, `cos`, `exp`, and `log`

Use unqualified math calls after `using std::sin`, `using std::cos`, or
`using std::exp`/`using std::log`. This supports ordinary scalars now and
enables argument-dependent lookup to find dual-number overloads later.

Checkpoint questions:

- Why is `Sine<E>` a different type from `Cosine<E>` even if their storage is
  identical?
- Why does an unqualified dependent call matter for later extensibility?
- Which operations can participate in constant evaluation under the selected
  compiler and C++20 standard-library implementation?

## Milestone 4: Symbolic Differentiation

Implement `include/expr_ad/differentiate.hpp` against
`tests/test_differentiate.cpp`:

- primary template `derivative<E, WithRespectTo>`
- convenience alias `derivative_t`
- `differentiate<Index>(expression)`
- partial specializations for every leaf, binary node, and unary function

`E` is the expression type, while `WithRespectTo` is a concrete variable-node
type such as `Variable<0>`. To express that contract explicitly, an
implementation may define a concept that recognizes only variable nodes:

```cpp
template <typename T>
struct is_variable : std::false_type {};

template <std::size_t Index>
struct is_variable<Variable<Index>> : std::true_type {};

template <typename T>
inline constexpr bool is_variable_v =
    is_variable<std::remove_cvref_t<T>>::value;

template <typename T>
concept VariableNode = Expression<T> && is_variable_v<T>;

template <Expression E, VariableNode WithRespectTo>
struct derivative;
```

`VariableNode` is a compile-time predicate; `WithRespectTo` remains a type
parameter. The lowercase `derivative` type trait recursively deduces the
expression-tree type representing the symbolic derivative, following the
standard type-trait naming convention such as `std::remove_reference`.
`derivative_t` is the shorthand for the resulting nested `type`.

Each specialization computes an output `type` and provides `to_derivative()`
to construct that output from the runtime values in the original tree.
Implement the sum, product, quotient, negation, sine, cosine, exponential, and
logarithm rules.

### The role of `to_derivative()`

`derivative_t<E, WithRespectTo>` computes only the derivative's expression-tree
type. A type can record that an operand is a `Constant<double>`, but it cannot
record that a particular object stores `3.0`. `to_derivative()` bridges that
gap with the following contract:

```cpp
static constexpr auto to_derivative(const ExpressionType& expression) -> type;
```

The input is the original expression node, passed by `const` reference. It
contains the child expressions and any runtime values stored in them. The
variable with respect to which the expression is differentiated is not a
function argument; it is encoded in the `derivative` specialization. The output
is a new symbolic derivative tree of the nested `type`, returned by value.

For example, the product rule recursively constructs both child derivatives
and copies the original operands that the rule must retain:

```cpp
template <Expression L, Expression R, std::size_t VariableIndex>
struct derivative<Multiply<L, R>, Variable<VariableIndex>> {
  using ExpressionType = Multiply<L, R>;
  using VariableType = Variable<VariableIndex>;
  using type = Add<Multiply<derivative_t<L, VariableType>, R>,
                   Multiply<L, derivative_t<R, VariableType>>>;

  static constexpr auto to_derivative(const ExpressionType& expression)
      -> type {
    return {
        {derivative<L, VariableType>::to_derivative(expression.left),
         expression.right},
        {expression.left,
         derivative<R, VariableType>::to_derivative(expression.right)}};
  }
};
```

For `x * 3.0`, `type` describes the unsimplified symbolic expression
`1 * 3.0 + x * 0`, while `to_derivative()` copies the stored value `3.0` into
that new tree. It neither evaluates nor simplifies the result. Evaluation
remains a separate step, as does simplification after completing milestone 5:

```cpp
const auto raw_derivative = differentiate<0>(x * 3.0);
const auto reduced_derivative = simplify(raw_derivative);
const auto value = evaluate(reduced_derivative, 2.0); // 3.0
```

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
- `to_simplified(expression)` on each `simplifier` specialization
- recursive child simplification
- `0 + e -> e`, `e + 0 -> e`
- `1 * e -> e`, `e * 1 -> e`, and `0 * e -> 0`
- `e - 0 -> e`, `e / 1 -> e`, and `-(-e) -> e`

Keep the rules local and deterministic. This milestone is not a general
computer-algebra system and should not reorder terms or combine like terms.

### The role of `to_simplified()`

`simplified_t<E>` computes only the simplified expression-tree type. As with
symbolic differentiation, that type cannot retain object state such as the
value stored in a `Constant<double>`. Each `simplifier` specialization therefore
uses this contract:

```cpp
static constexpr auto to_simplified(const ExpressionType& expression) -> type;
```

The input is the original expression node, passed by `const` reference. The
output is a new expression tree of the nested `type`, returned by value with
runtime constant values preserved. Composite nodes first simplify their
children, then select and apply a local rewrite to those simplified children:

```cpp
template <Expression L, Expression R>
struct simplifier<Add<L, R>> {
  using LeftRule = simplifier<L>;
  using RightRule = simplifier<R>;
  using Left = typename LeftRule::type;
  using Right = typename RightRule::type;
  using Rewrite = add_rule<Left, Right>;
  using type = typename Rewrite::type;

  static constexpr auto to_simplified(const Add<L, R>& expression) -> type {
    return Rewrite::make(LeftRule::to_simplified(expression.left),
                         RightRule::to_simplified(expression.right));
  }
};
```

Here, `to_simplified()` performs the recursive tree transformation, while the
local `Rewrite::make()` only constructs the selected parent result from children
that are already simplified. For example:

```cpp
const auto original = (x + 2.0) + Zero{};
const auto reduced = simplify(original);
const auto value = evaluate(reduced, 3.0); // 5.0
```

The rewrite removes the outer `+ Zero{}`, but the stored value `2.0` survives in
the resulting `Constant<double>` object.

Checkpoint questions:

- Why must children be simplified before selecting the parent rule?
- Why can a runtime-valued `Constant<T>` not drive type-level zero elimination?
- Which rewrites change floating-point behavior for NaN, infinity, signed zero,
  or exceptions?

## Milestone 6: Forward-Mode AD

Implement `include/expr_ad/dual.hpp` against `tests/test_forward_ad.cpp`:

- `Dual<T>{value, derivative}`
- addition, subtraction, multiplication, division, and unary negation
- dual overloads of `sin`, `cos`, `exp`, and `log`

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
- Real-valued logarithms require a positive operand; the expression type does
  not track or enforce that domain.
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
