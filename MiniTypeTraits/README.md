# Mini Type Traits

Build a small `<type_traits>`-style library in namespace `meta`. The headers are
intentionally empty scaffolds, while the tests are concrete compile-time
specifications that compare your implementation with the standard library.

An older partial implementation exists in
`CppTemplatesCompleteGuide/scratch/include/traits.hpp`. Avoid consulting it
until you have completed a milestone if you want to solve each exercise
independently.

## Learning Loop

For each milestone:

1. Read the relevant test and predict which assertions should pass.
2. Implement the primary template before any specialization.
3. Add specializations one at a time and explain why each pattern matches.
4. Configure CMake with that milestone enabled.
5. Treat the first compiler error as the next missing behavior.
6. Compare your implementation with `<type_traits>`, not just the expected
   `true` or `false` value.
7. Do not proceed until you can explain the checkpoint questions.

The tests have empty `main` functions because all verification happens through
`static_assert` during compilation.

## Build

Milestone 0 verifies only that the empty umbrella header can be included:

```sh
cmake -S . -B build -DMINI_TRAITS_MILESTONE=0
cmake --build build --target mini_traits_scaffold
```

Select the milestone you are currently implementing:

```sh
cmake -S . -B build -DMINI_TRAITS_MILESTONE=1
cmake --build build --target mini_traits_foundation
```

After milestone 5, build every MiniTypeTraits test and run CTest:

```sh
cmake -S . -B build -DMINI_TRAITS_MILESTONE=5
cmake --build build -j
ctest --test-dir build -R mini_traits --output-on-failure
```

Increasing the milestone enables that milestone and all earlier ones. A newly
enabled milestone is expected to fail compilation until its traits are done.

## Milestone 1: Trait Mechanics

Files:

- `include/meta/integral_constant.hpp`
- `include/meta/comparison.hpp`
- `include/meta/cv.hpp`
- `include/meta/reference.hpp`
- `tests/test_foundation.cpp`

Implement in this order:

1. `integral_constant`, `bool_constant`, `true_type`, `false_type`
2. `is_same` and `is_same_v`
3. `remove_const` and `remove_const_t`
4. `remove_reference` and `remove_reference_t`
5. `is_const` and `is_const_v`
6. `is_lvalue_reference` and `is_lvalue_reference_v`
7. `is_rvalue_reference` and `is_rvalue_reference_v`
8. `is_reference` and `is_reference_v`

Checkpoint questions:

- Why does `is_same<T, U>` default to false?
- Why is `is_same<T, T>` more specialized?
- Why are `const int` and `const int&` different for `is_const`?
- Why do transformation traits expose `type`, while predicates expose `value`?
- What convenience do `_t` and `_v` provide without adding a new mechanism?

## Milestone 2: Composition And Selection

Files:

- `include/meta/cv.hpp`
- `include/meta/reference.hpp`
- `include/meta/pointer.hpp`
- `include/meta/conditional.hpp`
- `include/meta/arithmetic.hpp`
- `tests/test_transformations.cpp`
- `tests/test_predicates.cpp`
- `tests/test_selection_arithmetic.cpp`

Implement:

- `remove_volatile`, `remove_cv`, `is_volatile`
- `remove_pointer`, `is_pointer`
- `add_const`, `add_lvalue_reference`, `add_rvalue_reference`, `add_pointer`
- `is_void`, `is_integral`, `is_floating_point`, `is_arithmetic`
- `conditional`, `enable_if`
- Every corresponding `_t` or `_v` helper used by the tests

Normalize cv-qualified input before applying raw-type helpers. Pay special
attention to the difference between `const int*` and `int* const`, and to the
fact that `void&` cannot be formed. Implementing `add_lvalue_reference<void>`
correctly is the first useful SFINAE exercise.

Checkpoint questions:

- Why does removing cv-qualification from `const int*` leave `const int*`?
- Why should `is_pointer<int* const>` be true but `is_pointer<int*&>` false?
- How does `conditional` select a type without instantiating both alternatives?
- Why does the false `enable_if` specialization deliberately omit `type`?

## Milestone 3: Arrays And Variadic Logic

Files:

- `include/meta/array.hpp`
- `include/meta/logical.hpp`
- `tests/test_arrays.cpp`
- `tests/test_logical.cpp`

Implement:

- `is_array`, including known and unknown bounds
- `remove_extent` and `remove_all_extents`
- `rank` and `extent`
- `conjunction`, `disjunction`, and `negation`
- Every corresponding `_t` or `_v` helper used by the tests

The logical traits must short-circuit. The incomplete-type assertions ensure
that an unused later operand is not instantiated.

Checkpoint questions:

- How does recursive partial specialization peel one array dimension?
- What should `extent<T, I>` return when `I` is beyond the array rank?
- Why is inheriting from the decisive operand different from merely computing
  a Boolean expression over every operand?

## Milestone 4: Detection And Expression SFINAE

Files:

- `include/meta/detection.hpp`
- `tests/test_detection.cpp`

Implement:

- `void_t`
- `has_value_type` and `has_value_type_v`
- `has_begin` and `has_begin_v`
- `is_addable<T, U>` and `is_addable_v<T, U>`

Use `decltype` and `std::declval` to describe expressions without evaluating
them. Include `<utility>` in your implementation rather than depending on a
test's transitive includes.

Checkpoint questions:

- Which substitution fails when `T::value_type` does not exist?
- Why does `std::declval<T&>()` need no definition?
- Does `is_addable<T, U>` ask whether the result has a particular type, or only
  whether the expression is well-formed?

## Milestone 5: Realistic Transformations

Files:

- `include/meta/transformations.hpp`
- `tests/test_realistic_transformations.cpp`

Implement:

- `remove_cvref` and `remove_cvref_t`
- `decay` and `decay_t`

Build `remove_cvref` from earlier traits. For `decay`, first remove references,
then distinguish arrays, functions, and ordinary types. Function detection may
use `std::is_function_v` initially; implementing a complete `is_function` is a
separate advanced exercise.

Checkpoint questions:

- Why must `decay` inspect the type after removing references?
- Why does an array become a pointer to its element type?
- Why does a function type become a function pointer?
- How does `decay<const int&>` differ from only removing the reference?

## Optional Concepts Exercise

After all tests pass, define concepts in a separate experiment rather than in
the traits library:

```cpp
template <class T>
concept Integral = meta::is_integral_v<T>;

template <class T>
concept Pointer = meta::is_pointer_v<T>;
```

Write one overload constrained with `meta::enable_if_t` and an equivalent one
using `requires`. Compare their declarations and their compiler diagnostics.

## Testing Checklist

Every new trait should cover:

- A positive case
- A negative case
- Top-level `const` and `volatile` where meaningful
- Lvalue and rvalue references where meaningful
- An edge case such as `void`, an unknown-bound array, or a function type
- Equality with the corresponding standard trait

Do not add declarations to namespace `std`. The implementation belongs only in
namespace `meta`.
