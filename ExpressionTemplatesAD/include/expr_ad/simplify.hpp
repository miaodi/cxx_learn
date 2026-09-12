#pragma once

#include "expr_ad/differentiate.hpp"

namespace expr_ad {

// Milestone 5: implement simplifier<Expression>, simplified_t, and simplify().
// Recursively apply the local Zero, One, and double-negation identities tested
// in tests/test_simplify.cpp.

// Primary interface
template <Expression E>
struct simplifier {
  using type = E;
  static constexpr type to_simplified(const E &expr) { return expr; }
};

template <Expression E>
using simplified_t = typename simplifier<std::remove_cvref_t<E>>::type;

// Unary nodes
template <Expression E>
struct simplifier<Negate<Negate<E>>> {
  using type = typename simplifier<E>::type;
  static constexpr type to_simplified(const Negate<Negate<E>> &neg) {
    return simplifier<E>::to_simplified(neg.expr.expr);
  }
};

template <Expression E>
struct simplifier<Logarithm<E>> {
  using type = Logarithm<typename simplifier<E>::type>;
  static constexpr type to_simplified(const Logarithm<E> &log) {
    return type{simplifier<E>::to_simplified(log.expr)};
  }
};
// Binary nodes
template <Expression L, Expression R>
struct simplifier<Add<L, R>> {
  using LeftSimplified = typename simplifier<L>::type;
  using RightSimplified = typename simplifier<R>::type;
  static constexpr bool is_left_zero = std::is_same_v<LeftSimplified, Zero>;
  static constexpr bool is_right_zero = std::is_same_v<RightSimplified, Zero>;
  using type = std::conditional_t<
      is_left_zero, RightSimplified,
      std::conditional_t<is_right_zero, LeftSimplified,
                         Add<LeftSimplified, RightSimplified>>>;
  static constexpr type to_simplified(const Add<L, R> &add) {
    if constexpr (is_left_zero) {
      return simplifier<R>::to_simplified(add.right);
    } else if constexpr (is_right_zero) {
      return simplifier<L>::to_simplified(add.left);
    } else {
      return type{simplifier<L>::to_simplified(add.left),
                  simplifier<R>::to_simplified(add.right)};
    }
  }
};

template <Expression L, Expression R>
struct simplifier<Subtract<L, R>> {
  using LeftSimplified = typename simplifier<L>::type;
  using RightSimplified = typename simplifier<R>::type;
  static constexpr bool is_left_zero = std::is_same_v<LeftSimplified, Zero>;
  static constexpr bool is_right_zero = std::is_same_v<RightSimplified, Zero>;
  using type = std::conditional_t<
      is_left_zero, Negate<RightSimplified>,
      std::conditional_t<is_right_zero, LeftSimplified,
                         Subtract<LeftSimplified, RightSimplified>>>;
  static constexpr type to_simplified(const Subtract<L, R> &sub) {
    if constexpr (is_left_zero) {
      return type{simplifier<R>::to_simplified(sub.right)};
    } else if constexpr (is_right_zero) {
      return simplifier<L>::to_simplified(sub.left);
    } else {
      return type{simplifier<L>::to_simplified(sub.left),
                  simplifier<R>::to_simplified(sub.right)};
    }
  }
};

template <Expression L, Expression R>
struct simplifier<Multiply<L, R>> {

  using LeftSimplified = typename simplifier<L>::type;
  using RightSimplified = typename simplifier<R>::type;
  static constexpr bool is_left_zero = std::is_same_v<LeftSimplified, Zero>;
  static constexpr bool is_right_zero = std::is_same_v<RightSimplified, Zero>;
  static constexpr bool is_left_one = std::is_same_v<LeftSimplified, One>;
  static constexpr bool is_right_one = std::is_same_v<RightSimplified, One>;

  using type = std::conditional_t<
      is_left_zero || is_right_zero, Zero,
      std::conditional_t<
          is_left_one, RightSimplified,
          std::conditional_t<is_right_one, LeftSimplified,
                             Multiply<LeftSimplified, RightSimplified>>>>;
  static constexpr type to_simplified(const Multiply<L, R> &mul) {
    if constexpr (is_left_zero || is_right_zero) {
      return type{};
    } else if constexpr (is_left_one) {
      return simplifier<R>::to_simplified(mul.right);
    } else if constexpr (is_right_one) {
      return simplifier<L>::to_simplified(mul.left);
    } else {
      return type{simplifier<L>::to_simplified(mul.left),
                  simplifier<R>::to_simplified(mul.right)};
    }
  }
};

template <Expression L, Expression R>
struct simplifier<Divide<L, R>> {
  using LeftSimplified = typename simplifier<L>::type;
  using RightSimplified = typename simplifier<R>::type;
  static constexpr bool is_left_zero = std::is_same_v<LeftSimplified, Zero>;
  static constexpr bool is_right_zero = std::is_same_v<RightSimplified, Zero>;
  static constexpr bool is_left_one = std::is_same_v<LeftSimplified, One>;
  static constexpr bool is_right_one = std::is_same_v<RightSimplified, One>;

  static_assert(!is_right_zero, "Division by zero is undefined.");
  using type = std::conditional_t<
      is_right_one, LeftSimplified,
      std::conditional_t<is_left_zero, Zero,
                         Divide<LeftSimplified, RightSimplified>>>;
  static constexpr type to_simplified(const Divide<L, R> &div) {
    if constexpr (is_right_one) {
      return simplifier<L>::to_simplified(div.left);
    } else if constexpr (is_left_zero) {
      return type{};
    } else {
      return type{simplifier<L>::to_simplified(div.left),
                  simplifier<R>::to_simplified(div.right)};
    }
  }
};

template <Expression E>
static constexpr simplified_t<E> simplify(const E &expr) {
  return simplifier<std::remove_cvref_t<E>>::to_simplified(expr);
}
} // namespace expr_ad
