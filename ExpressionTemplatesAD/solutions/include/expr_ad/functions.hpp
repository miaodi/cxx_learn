#pragma once

#include "expr_ad/operators.hpp"

#include <cmath>
#include <type_traits>
#include <utility>

namespace expr_ad {

template <Expression Left, Expression Right>
struct Subtract {
  using expression_tag = void;

  Left left;
  Right right;

  template <typename... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) - right.evaluate(values...);
  }
};

template <Expression Left, Expression Right>
struct Divide {
  using expression_tag = void;

  Left left;
  Right right;

  template <typename... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) / right.evaluate(values...);
  }
};

template <Expression Operand>
struct Negate {
  using expression_tag = void;

  Operand operand;

  template <typename... Values>
  constexpr auto evaluate(const Values &...values) const {
    return -operand.evaluate(values...);
  }
};

template <Expression Operand>
struct Sine {
  using expression_tag = void;

  Operand operand;

  template <typename... Values>
  constexpr auto evaluate(const Values &...values) const {
    using std::sin;
    return sin(operand.evaluate(values...));
  }
};

template <Expression Operand>
struct Cosine {
  using expression_tag = void;

  Operand operand;

  template <typename... Values>
  constexpr auto evaluate(const Values &...values) const {
    using std::cos;
    return cos(operand.evaluate(values...));
  }
};

template <Expression Operand>
struct Exponential {
  using expression_tag = void;

  Operand operand;

  template <typename... Values>
  constexpr auto evaluate(const Values &...values) const {
    using std::exp;
    return exp(operand.evaluate(values...));
  }
};

template <ExpressionOperand L, ExpressionOperand R>
  requires(Expression<L> || Expression<R>)
constexpr auto operator-(L &&left, R &&right) {
  auto stored_left = as_expression(std::forward<L>(left));
  auto stored_right = as_expression(std::forward<R>(right));
  return Subtract<decltype(stored_left), decltype(stored_right)>{
      std::move(stored_left), std::move(stored_right)};
}

template <ExpressionOperand L, ExpressionOperand R>
  requires(Expression<L> || Expression<R>)
constexpr auto operator/(L &&left, R &&right) {
  auto stored_left = as_expression(std::forward<L>(left));
  auto stored_right = as_expression(std::forward<R>(right));
  return Divide<decltype(stored_left), decltype(stored_right)>{
      std::move(stored_left), std::move(stored_right)};
}

template <Expression E>
constexpr auto operator-(E &&expression) {
  using Stored = std::remove_cvref_t<E>;
  return Negate<Stored>{std::forward<E>(expression)};
}

template <Expression E>
constexpr auto sin(E &&expression) {
  using Stored = std::remove_cvref_t<E>;
  return Sine<Stored>{std::forward<E>(expression)};
}

template <Expression E>
constexpr auto cos(E &&expression) {
  using Stored = std::remove_cvref_t<E>;
  return Cosine<Stored>{std::forward<E>(expression)};
}

template <Expression E>
constexpr auto exp(E &&expression) {
  using Stored = std::remove_cvref_t<E>;
  return Exponential<Stored>{std::forward<E>(expression)};
}

} // namespace expr_ad
