#pragma once

#include "expr_ad/core.hpp"

#include <type_traits>
#include <utility>

namespace expr_ad {

template <Expression Left, Expression Right>
struct Add {
  using expression_tag = void;

  Left left;
  Right right;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) + right.evaluate(values...);
  }
};

template <Expression Left, Expression Right>
struct Subtract {
  using expression_tag = void;

  Left left;
  Right right;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) - right.evaluate(values...);
  }
};

template <Expression Left, Expression Right>
struct Multiply {
  using expression_tag = void;

  Left left;
  Right right;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) * right.evaluate(values...);
  }
};

template <Expression Left, Expression Right>
struct Divide {
  using expression_tag = void;

  Left left;
  Right right;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) / right.evaluate(values...);
  }
};

template <ArithmeticValue T>
constexpr auto constant(T &&value) {
  using Value = std::remove_cvref_t<T>;
  return Constant<Value>{std::forward<T>(value)};
}

template <typename T>
concept ExpressionOperand = Expression<T> || ArithmeticValue<T>;

template <ExpressionOperand T>
constexpr auto as_expression(T &&value) {
  if constexpr (Expression<T>) {
    return std::forward<T>(value);
  } else {
    return constant(std::forward<T>(value));
  }
}

template <ExpressionOperand L, ExpressionOperand R>
  requires(Expression<L> || Expression<R>)
constexpr auto operator+(L &&left, R &&right) {
  auto stored_left = as_expression(std::forward<L>(left));
  auto stored_right = as_expression(std::forward<R>(right));
  return Add<decltype(stored_left), decltype(stored_right)>{
      std::move(stored_left), std::move(stored_right)};
}

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
constexpr auto operator*(L &&left, R &&right) {
  auto stored_left = as_expression(std::forward<L>(left));
  auto stored_right = as_expression(std::forward<R>(right));
  return Multiply<decltype(stored_left), decltype(stored_right)>{
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

} // namespace expr_ad
