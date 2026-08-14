#pragma once

#include "expr_ad/functions.hpp"

#include <cstddef>
#include <type_traits>

namespace expr_ad {

template <typename...>
inline constexpr bool always_false_v = false;

template <typename E, typename VariableType>
struct derivative {
  static_assert(always_false_v<E>,
                "No symbolic derivative rule exists for this expression node");
};

template <typename E, typename VariableType>
using derivative_t =
    typename derivative<std::remove_cvref_t<E>,
                        std::remove_cvref_t<VariableType>>::type;

template <typename T, std::size_t VariableIndex>
struct derivative<Constant<T>, Variable<VariableIndex>> {
  using type = Zero;

  static constexpr type make(const Constant<T> &) { return {}; }
};

template <std::size_t VariableIndex>
struct derivative<Zero, Variable<VariableIndex>> {
  using type = Zero;

  static constexpr type make(const Zero &) { return {}; }
};

template <std::size_t VariableIndex>
struct derivative<One, Variable<VariableIndex>> {
  using type = Zero;

  static constexpr type make(const One &) { return {}; }
};

template <std::size_t Index, std::size_t VariableIndex>
struct derivative<Variable<Index>, Variable<VariableIndex>> {
  using type = std::conditional_t<Index == VariableIndex, One, Zero>;

  static constexpr type make(const Variable<Index> &) { return {}; }
};

template <Expression L, Expression R, std::size_t VariableIndex>
struct derivative<Add<L, R>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type =
      Add<derivative_t<L, VariableType>, derivative_t<R, VariableType>>;

  static constexpr type make(const Add<L, R> &expression) {
    return {derivative<L, VariableType>::make(expression.left),
            derivative<R, VariableType>::make(expression.right)};
  }
};

template <Expression L, Expression R, std::size_t VariableIndex>
struct derivative<Subtract<L, R>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type =
      Subtract<derivative_t<L, VariableType>, derivative_t<R, VariableType>>;

  static constexpr type make(const Subtract<L, R> &expression) {
    return {derivative<L, VariableType>::make(expression.left),
            derivative<R, VariableType>::make(expression.right)};
  }
};

template <Expression L, Expression R, std::size_t VariableIndex>
struct derivative<Multiply<L, R>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Add<Multiply<derivative_t<L, VariableType>, R>,
                   Multiply<L, derivative_t<R, VariableType>>>;

  static constexpr type make(const Multiply<L, R> &expression) {
    return {
        {derivative<L, VariableType>::make(expression.left), expression.right},
        {expression.left, derivative<R, VariableType>::make(expression.right)}};
  }
};

template <Expression L, Expression R, std::size_t VariableIndex>
struct derivative<Divide<L, R>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using Numerator = Subtract<Multiply<derivative_t<L, VariableType>, R>,
                             Multiply<L, derivative_t<R, VariableType>>>;
  using Denominator = Multiply<R, R>;
  using type = Divide<Numerator, Denominator>;

  static constexpr type make(const Divide<L, R> &expression) {
    return {
        {{derivative<L, VariableType>::make(expression.left), expression.right},
         {expression.left,
          derivative<R, VariableType>::make(expression.right)}},
        {expression.right, expression.right}};
  }
};

template <Expression E, std::size_t VariableIndex>
struct derivative<Negate<E>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Negate<derivative_t<E, VariableType>>;

  static constexpr type make(const Negate<E> &expression) {
    return {derivative<E, VariableType>::make(expression.operand)};
  }
};

template <Expression E, std::size_t VariableIndex>
struct derivative<Sine<E>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Multiply<Cosine<E>, derivative_t<E, VariableType>>;

  static constexpr type make(const Sine<E> &expression) {
    return {{expression.operand},
            derivative<E, VariableType>::make(expression.operand)};
  }
};

template <Expression E, std::size_t VariableIndex>
struct derivative<Cosine<E>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Multiply<Negate<Sine<E>>, derivative_t<E, VariableType>>;

  static constexpr type make(const Cosine<E> &expression) {
    return {{{expression.operand}},
            derivative<E, VariableType>::make(expression.operand)};
  }
};

template <Expression E, std::size_t VariableIndex>
struct derivative<Exponential<E>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Multiply<Exponential<E>, derivative_t<E, VariableType>>;

  static constexpr type make(const Exponential<E> &expression) {
    return {{expression.operand},
            derivative<E, VariableType>::make(expression.operand)};
  }
};

template <std::size_t VariableIndex, Expression E>
constexpr auto differentiate(const E &expression) {
  using ExpressionType = std::remove_cvref_t<E>;
  using VariableType = Variable<VariableIndex>;
  return derivative<ExpressionType, VariableType>::make(expression);
}

} // namespace expr_ad
