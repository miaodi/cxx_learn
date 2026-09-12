#pragma once

#include "expr_ad/core.hpp"

#include <type_traits>
#include <utility>

namespace expr_ad {

// Milestone 2: implement the binary expression vocabulary and builders: Add,
// Subtract, Multiply, Divide, constant(value), ExpressionOperand,
// value-owning conversion to an expression, and constrained +, -, *, and /.
template <Expression L, Expression R>
struct Add {
  using expression_tag = void;

  L left;
  R right;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) + right.evaluate(values...);
  }
};

template <Expression L, Expression R>
struct Subtract {
  using expression_tag = void;

  L left;
  R right;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) - right.evaluate(values...);
  }
};

template <Expression L, Expression R>
struct Multiply {
  using expression_tag = void;

  L left;
  R right;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) * right.evaluate(values...);
  }
};

template <Expression L, Expression R>
struct Divide {
  using expression_tag = void;

  L left;
  R right;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) / right.evaluate(values...);
  }
};

template <typename T>
concept ExpressionOperand = Expression<T> || ArithmeticValue<T>;

template <ExpressionOperand T>
constexpr auto as_expression(T &&value) {
  if constexpr (Expression<std::remove_cvref_t<T>>) {
    return std::forward<T>(value);
  } else {
    return Constant<std::remove_cvref_t<T>>{std::forward<T>(value)};
  }
}

template <ExpressionOperand L, ExpressionOperand R>
constexpr auto operator+(L &&left, R &&right) {
  return Add{as_expression(std::forward<L>(left)),
             as_expression(std::forward<R>(right))};
}

template <ExpressionOperand L, ExpressionOperand R>
constexpr auto operator-(L &&left, R &&right) {
  return Subtract{as_expression(std::forward<L>(left)),
                  as_expression(std::forward<R>(right))};
}

template <ExpressionOperand L, ExpressionOperand R>
constexpr auto operator*(L &&left, R &&right) {
  return Multiply{as_expression(std::forward<L>(left)),
                  as_expression(std::forward<R>(right))};
}
template <ExpressionOperand L, ExpressionOperand R>
constexpr auto operator/(L &&left, R &&right) {
  return Divide{as_expression(std::forward<L>(left)),
                as_expression(std::forward<R>(right))};
}

} // namespace expr_ad
