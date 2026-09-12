#pragma once

#include "expr_ad/binary.hpp"
#include "expr_ad/unary.hpp"

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

  static constexpr type to_derivative(const Constant<T> &) { return {}; }
};

template <std::size_t VariableIndex>
struct derivative<Zero, Variable<VariableIndex>> {
  using type = Zero;

  static constexpr type to_derivative(const Zero &) { return {}; }
};

template <std::size_t VariableIndex>
struct derivative<One, Variable<VariableIndex>> {
  using type = Zero;

  static constexpr type to_derivative(const One &) { return {}; }
};

template <std::size_t Index, std::size_t VariableIndex>
struct derivative<Variable<Index>, Variable<VariableIndex>> {
  using type = std::conditional_t<Index == VariableIndex, One, Zero>;

  static constexpr type to_derivative(const Variable<Index> &) { return {}; }
};

template <Expression L, Expression R, std::size_t VariableIndex>
struct derivative<Add<L, R>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type =
      Add<derivative_t<L, VariableType>, derivative_t<R, VariableType>>;

  static constexpr type to_derivative(const Add<L, R> &expression) {
    return {derivative<L, VariableType>::to_derivative(expression.left),
            derivative<R, VariableType>::to_derivative(expression.right)};
  }
};

template <Expression L, Expression R, std::size_t VariableIndex>
struct derivative<Subtract<L, R>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type =
      Subtract<derivative_t<L, VariableType>, derivative_t<R, VariableType>>;

  static constexpr type to_derivative(const Subtract<L, R> &expression) {
    return {derivative<L, VariableType>::to_derivative(expression.left),
            derivative<R, VariableType>::to_derivative(expression.right)};
  }
};

template <Expression L, Expression R, std::size_t VariableIndex>
struct derivative<Multiply<L, R>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Add<Multiply<derivative_t<L, VariableType>, R>,
                   Multiply<L, derivative_t<R, VariableType>>>;

  static constexpr type to_derivative(const Multiply<L, R> &expression) {
    return {
        {derivative<L, VariableType>::to_derivative(expression.left),
         expression.right},
        {expression.left,
         derivative<R, VariableType>::to_derivative(expression.right)}};
  }
};

template <Expression L, Expression R, std::size_t VariableIndex>
struct derivative<Divide<L, R>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using Numerator = Subtract<Multiply<derivative_t<L, VariableType>, R>,
                             Multiply<L, derivative_t<R, VariableType>>>;
  using Denominator = Multiply<R, R>;
  using type = Divide<Numerator, Denominator>;

  static constexpr type to_derivative(const Divide<L, R> &expression) {
    return {
        {{derivative<L, VariableType>::to_derivative(expression.left),
          expression.right},
         {expression.left,
          derivative<R, VariableType>::to_derivative(expression.right)}},
        {expression.right, expression.right}};
  }
};

template <Expression E, std::size_t VariableIndex>
struct derivative<Negate<E>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Negate<derivative_t<E, VariableType>>;

  static constexpr type to_derivative(const Negate<E> &expression) {
    return {derivative<E, VariableType>::to_derivative(expression.operand)};
  }
};

template <Expression E, std::size_t VariableIndex>
struct derivative<Sine<E>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Multiply<Cosine<E>, derivative_t<E, VariableType>>;

  static constexpr type to_derivative(const Sine<E> &expression) {
    return {{expression.operand},
            derivative<E, VariableType>::to_derivative(expression.operand)};
  }
};

template <Expression E, std::size_t VariableIndex>
struct derivative<Cosine<E>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Multiply<Negate<Sine<E>>, derivative_t<E, VariableType>>;

  static constexpr type to_derivative(const Cosine<E> &expression) {
    return {{{expression.operand}},
            derivative<E, VariableType>::to_derivative(expression.operand)};
  }
};

template <Expression E, std::size_t VariableIndex>
struct derivative<Exponential<E>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Multiply<Exponential<E>, derivative_t<E, VariableType>>;

  static constexpr type to_derivative(const Exponential<E> &expression) {
    return {{expression.operand},
            derivative<E, VariableType>::to_derivative(expression.operand)};
  }
};

template <Expression E, std::size_t VariableIndex>
struct derivative<Logarithm<E>, Variable<VariableIndex>> {
  using VariableType = Variable<VariableIndex>;
  using type = Divide<derivative_t<E, VariableType>, E>;

  static constexpr type to_derivative(const Logarithm<E> &expression) {
    return {derivative<E, VariableType>::to_derivative(expression.operand),
            expression.operand};
  }
};

template <std::size_t VariableIndex, Expression E>
constexpr auto differentiate(const E &expression) {
  using ExpressionType = std::remove_cvref_t<E>;
  using VariableType = Variable<VariableIndex>;
  return derivative<ExpressionType, VariableType>::to_derivative(expression);
}

} // namespace expr_ad
