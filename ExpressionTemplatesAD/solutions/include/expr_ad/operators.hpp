#pragma once

#include "expr_ad/core.hpp"

#include <type_traits>
#include <utility>

namespace expr_ad {

template <typename T>
  requires std::is_arithmetic_v<std::remove_cvref_t<T>>
constexpr auto constant(T &&value) {
  using Value = std::remove_cvref_t<T>;
  return Constant<Value>{std::forward<T>(value)};
}

inline constexpr Variable<0> x{};
inline constexpr Variable<1> y{};

template <typename T>
concept ArithmeticValue = std::is_arithmetic_v<std::remove_cvref_t<T>>;

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
constexpr auto operator*(L &&left, R &&right) {
  auto stored_left = as_expression(std::forward<L>(left));
  auto stored_right = as_expression(std::forward<R>(right));
  return Multiply<decltype(stored_left), decltype(stored_right)>{
      std::move(stored_left), std::move(stored_right)};
}

} // namespace expr_ad
