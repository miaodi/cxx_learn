#pragma once
#include "expr_ad/binary.hpp"
#include "expr_ad/core.hpp"
#include "expr_ad/unary.hpp"
#include <cstddef>

namespace expr_ad {

// Milestone 4: implement derivative<Expression, Variable>, derivative_t, and
// differentiate<Index>(expression) with partial specializations for every node.

template <Expression InputExpression, VariableNode WithRespectTo>
struct derivative;

template <Expression InputExpression, VariableNode WithRespectTo>
using derivative_t =
    typename derivative<std::remove_cvref_t<InputExpression>,
                        std::remove_cvref_t<WithRespectTo>>::type;

template <ArithmeticValue T, std::size_t Index>
struct derivative<Constant<T>, Variable<Index>> {
  using type = Zero;
  static constexpr type to_derivative(const Constant<T> &) { return Zero{}; }
};

template <std::size_t Index, std::size_t WithRespectToIndex>
struct derivative<Variable<Index>, Variable<WithRespectToIndex>> {
  using type = std::conditional_t<Index == WithRespectToIndex, One, Zero>;
  static constexpr type to_derivative(const Variable<Index> &) { return type{}; }
};

// Unary nodes (e.g., Negation) would have their derivative specializations
// here.
template <Expression E, VariableNode WithRespectTo>
struct derivative<Negate<E>, WithRespectTo> {
  using OperandRule = derivative<E, WithRespectTo>;
  using OperandDerivative = typename OperandRule::type;
  using type = Negate<OperandDerivative>;
  static constexpr type to_derivative(const Negate<E> &neg) {
    return type{OperandRule::to_derivative(neg.expr)};
  }
};

template <Expression E, VariableNode WithRespectTo>
struct derivative<Sine<E>, WithRespectTo> {
  using OperandRule = derivative<E, WithRespectTo>;
  using OperandDerivative = typename OperandRule::type;
  using type = Multiply<Cosine<E>, OperandDerivative>;
  static constexpr type to_derivative(const Sine<E> &sin) {
    return type{Cosine<E>{sin.expr}, OperandRule::to_derivative(sin.expr)};
  }
};

template <Expression E, VariableNode WithRespectTo>
struct derivative<Cosine<E>, WithRespectTo> {
  using OperandRule = derivative<E, WithRespectTo>;
  using OperandDerivative = typename OperandRule::type;
  using type = Multiply<Negate<Sine<E>>, OperandDerivative>;
  static constexpr type to_derivative(const Cosine<E> &cos) {
    return type{Negate<Sine<E>>{Sine<E>{cos.expr}},
                OperandRule::to_derivative(cos.expr)};
  }
};

template <Expression E, VariableNode WithRespectTo>
struct derivative<Exponential<E>, WithRespectTo> {
  using OperandRule = derivative<E, WithRespectTo>;
  using OperandDerivative = typename OperandRule::type;
  using type = Multiply<Exponential<E>, OperandDerivative>;
  static constexpr type to_derivative(const Exponential<E> &exp) {
    return type{Exponential<E>{exp.expr}, OperandRule::to_derivative(exp.expr)};
  }
};

template <Expression E, VariableNode WithRespectTo>
struct derivative<Logarithm<E>, WithRespectTo> {
  using OperandRule = derivative<E, WithRespectTo>;
  using OperandDerivative = typename OperandRule::type;
  using type = Divide<OperandDerivative, E>;
  static constexpr type to_derivative(const Logarithm<E> &log) {
    return type{OperandRule::to_derivative(log.expr), log.expr};
  }
};

// Binary nodes (e.g., Addition, Multiplication) would have their derivative
// specializations here.

template <Expression L, Expression R, VariableNode WithRespectTo>
struct derivative<Add<L, R>, WithRespectTo> {
  using LeftRule = derivative<L, WithRespectTo>;
  using RightRule = derivative<R, WithRespectTo>;
  using LeftDerivative = typename LeftRule::type;
  using RightDerivative = typename RightRule::type;
  using type = Add<LeftDerivative, RightDerivative>;
  static constexpr type to_derivative(const Add<L, R> &add) {
    return type{LeftRule::to_derivative(add.left),
                RightRule::to_derivative(add.right)};
  }
};

template <Expression L, Expression R, VariableNode WithRespectTo>
struct derivative<Subtract<L, R>, WithRespectTo> {
  using LeftRule = derivative<L, WithRespectTo>;
  using RightRule = derivative<R, WithRespectTo>;
  using LeftDerivative = typename LeftRule::type;
  using RightDerivative = typename RightRule::type;
  using type = Subtract<LeftDerivative, RightDerivative>;
  static constexpr type to_derivative(const Subtract<L, R> &sub) {
    return type{LeftRule::to_derivative(sub.left),
                RightRule::to_derivative(sub.right)};
  }
};

template <Expression L, Expression R, VariableNode WithRespectTo>
struct derivative<Multiply<L, R>, WithRespectTo> {
  using LeftRule = derivative<L, WithRespectTo>;
  using RightRule = derivative<R, WithRespectTo>;
  using LeftDerivative = typename LeftRule::type;
  using RightDerivative = typename RightRule::type;
  using type = Add<Multiply<LeftDerivative, R>, Multiply<L, RightDerivative>>;
  static constexpr type to_derivative(const Multiply<L, R> &mul) {
    return type{
        Multiply<LeftDerivative, R>{LeftRule::to_derivative(mul.left),
                                    mul.right},
        Multiply<L, RightDerivative>{mul.left,
                                     RightRule::to_derivative(mul.right)}};
  }
};

template <Expression L, Expression R, VariableNode WithRespectTo>
struct derivative<Divide<L, R>, WithRespectTo> {
  using LeftRule = derivative<L, WithRespectTo>;
  using RightRule = derivative<R, WithRespectTo>;
  using LeftDerivative = typename LeftRule::type;
  using RightDerivative = typename RightRule::type;
  using Numerator =
      Subtract<Multiply<LeftDerivative, R>, Multiply<L, RightDerivative>>;
  using Denominator = Multiply<R, R>;
  using type = Divide<Numerator, Denominator>;

  static constexpr type to_derivative(const Divide<L, R> &expression) {
    const auto left_derivative = LeftRule::to_derivative(expression.left);
    const auto right_derivative = RightRule::to_derivative(expression.right);

    return (left_derivative * expression.right -
            expression.left * right_derivative) /
           (expression.right * expression.right);
  }
};

template <std::size_t Index, Expression E>
constexpr auto differentiate(const E &expression) {
  return derivative<std::remove_cvref_t<decltype(expression)>,
                    Variable<Index>>::to_derivative(expression);
}
} // namespace expr_ad
