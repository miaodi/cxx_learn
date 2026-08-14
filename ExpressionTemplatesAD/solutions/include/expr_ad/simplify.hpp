#pragma once

#include "expr_ad/differentiate.hpp"

#include <type_traits>
#include <utility>

namespace expr_ad {

template <typename E>
struct simplifier {
  static_assert(always_false_v<E>,
                "No simplification rule exists for this expression node");
};

template <typename E>
using simplified_t = typename simplifier<std::remove_cvref_t<E>>::type;

template <Expression L, Expression R>
struct add_rule {
  static constexpr bool left_is_zero = std::is_same_v<L, Zero>;
  static constexpr bool right_is_zero = std::is_same_v<R, Zero>;
  using type =
      std::conditional_t<left_is_zero, R,
                         std::conditional_t<right_is_zero, L, Add<L, R>>>;

  static constexpr type make(L left, R right) {
    if constexpr (left_is_zero) {
      return right;
    } else if constexpr (right_is_zero) {
      return left;
    } else {
      return {std::move(left), std::move(right)};
    }
  }
};

template <Expression L, Expression R>
struct multiply_rule {
  static constexpr bool has_zero =
      std::is_same_v<L, Zero> || std::is_same_v<R, Zero>;
  static constexpr bool left_is_one = std::is_same_v<L, One>;
  static constexpr bool right_is_one = std::is_same_v<R, One>;
  using type = std::conditional_t<
      has_zero, Zero,
      std::conditional_t<left_is_one, R,
                         std::conditional_t<right_is_one, L, Multiply<L, R>>>>;

  static constexpr type make(L left, R right) {
    if constexpr (has_zero) {
      return {};
    } else if constexpr (left_is_one) {
      return right;
    } else if constexpr (right_is_one) {
      return left;
    } else {
      return {std::move(left), std::move(right)};
    }
  }
};

template <Expression L, Expression R>
struct subtract_rule {
  static constexpr bool right_is_zero = std::is_same_v<R, Zero>;
  using type = std::conditional_t<right_is_zero, L, Subtract<L, R>>;

  static constexpr type make(L left, R right) {
    if constexpr (right_is_zero) {
      return left;
    } else {
      return {std::move(left), std::move(right)};
    }
  }
};

template <Expression L, Expression R>
struct divide_rule {
  static constexpr bool right_is_one = std::is_same_v<R, One>;
  using type = std::conditional_t<right_is_one, L, Divide<L, R>>;

  static constexpr type make(L left, R right) {
    if constexpr (right_is_one) {
      return left;
    } else {
      return {std::move(left), std::move(right)};
    }
  }
};

template <Expression E>
struct negate_rule {
  using type = Negate<E>;

  static constexpr type make(E expression) { return {std::move(expression)}; }
};

template <>
struct negate_rule<Zero> {
  using type = Zero;

  static constexpr type make(Zero) { return {}; }
};

template <Expression E>
struct negate_rule<Negate<E>> {
  using type = E;

  static constexpr type make(Negate<E> expression) {
    return std::move(expression.operand);
  }
};

template <typename T>
struct simplifier<Constant<T>> {
  using type = Constant<T>;

  static constexpr type make(const Constant<T> &expression) {
    return expression;
  }
};

template <std::size_t Index>
struct simplifier<Variable<Index>> {
  using type = Variable<Index>;

  static constexpr type make(const Variable<Index> &expression) {
    return expression;
  }
};

template <>
struct simplifier<Zero> {
  using type = Zero;

  static constexpr type make(const Zero &) { return {}; }
};

template <>
struct simplifier<One> {
  using type = One;

  static constexpr type make(const One &) { return {}; }
};

template <Expression L, Expression R>
struct simplifier<Add<L, R>> {
  using Left = simplified_t<L>;
  using Right = simplified_t<R>;
  using Rule = add_rule<Left, Right>;
  using type = typename Rule::type;

  static constexpr type make(const Add<L, R> &expression) {
    return Rule::make(simplifier<L>::make(expression.left),
                      simplifier<R>::make(expression.right));
  }
};

template <Expression L, Expression R>
struct simplifier<Multiply<L, R>> {
  using Left = simplified_t<L>;
  using Right = simplified_t<R>;
  using Rule = multiply_rule<Left, Right>;
  using type = typename Rule::type;

  static constexpr type make(const Multiply<L, R> &expression) {
    return Rule::make(simplifier<L>::make(expression.left),
                      simplifier<R>::make(expression.right));
  }
};

template <Expression L, Expression R>
struct simplifier<Subtract<L, R>> {
  using Left = simplified_t<L>;
  using Right = simplified_t<R>;
  using Rule = subtract_rule<Left, Right>;
  using type = typename Rule::type;

  static constexpr type make(const Subtract<L, R> &expression) {
    return Rule::make(simplifier<L>::make(expression.left),
                      simplifier<R>::make(expression.right));
  }
};

template <Expression L, Expression R>
struct simplifier<Divide<L, R>> {
  using Left = simplified_t<L>;
  using Right = simplified_t<R>;
  using Rule = divide_rule<Left, Right>;
  using type = typename Rule::type;

  static constexpr type make(const Divide<L, R> &expression) {
    return Rule::make(simplifier<L>::make(expression.left),
                      simplifier<R>::make(expression.right));
  }
};

template <Expression E>
struct simplifier<Negate<E>> {
  using Operand = simplified_t<E>;
  using Rule = negate_rule<Operand>;
  using type = typename Rule::type;

  static constexpr type make(const Negate<E> &expression) {
    return Rule::make(simplifier<E>::make(expression.operand));
  }
};

template <Expression E>
struct simplifier<Sine<E>> {
  using Operand = simplified_t<E>;
  using type = Sine<Operand>;

  static constexpr type make(const Sine<E> &expression) {
    return {simplifier<E>::make(expression.operand)};
  }
};

template <Expression E>
struct simplifier<Cosine<E>> {
  using Operand = simplified_t<E>;
  using type = Cosine<Operand>;

  static constexpr type make(const Cosine<E> &expression) {
    return {simplifier<E>::make(expression.operand)};
  }
};

template <Expression E>
struct simplifier<Exponential<E>> {
  using Operand = simplified_t<E>;
  using type = Exponential<Operand>;

  static constexpr type make(const Exponential<E> &expression) {
    return {simplifier<E>::make(expression.operand)};
  }
};

template <Expression E>
constexpr auto simplify(const E &expression) {
  return simplifier<std::remove_cvref_t<E>>::make(expression);
}

} // namespace expr_ad
