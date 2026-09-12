#pragma once

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace expr_ad {

// Milestone 1: implement the expression marker trait/concept, numeric concepts,
// and the terminal nodes Constant<T>, Variable<Index>, Zero, and One. Give
// every node value semantics and a constexpr evaluate(values...) member.
template <typename T, typename = void>
struct is_expression : std::false_type {};

template <typename T>
struct is_expression<
    T, std::void_t<typename std::remove_cvref_t<T>::expression_tag>>
    : std::true_type {};

template <typename T>
inline constexpr bool is_expression_v = is_expression<T>::value;

template <typename T>
concept Expression = is_expression_v<T>;

template <typename T>
concept ArithmeticValue = std::is_arithmetic_v<std::remove_cvref_t<T>>;

template <typename T>
concept Numeric = std::copy_constructible<std::remove_cvref_t<T>> &&
                  requires(const std::remove_cvref_t<T> &left,
                           const std::remove_cvref_t<T> &right) {
                    left + right;
                    left - right;
                    left * right;
                    left / right;
                    -left;
                  };

template <ArithmeticValue T>
struct Constant {
  T value;
  typedef T value_type;
  using expression_tag = void;
  template <Numeric... Values>
  constexpr T evaluate(const Values &...) const {
    return value;
  }
};

struct One {
  using expression_tag = void;
  template <Numeric... Values>
  constexpr int evaluate(const Values &...) const {
    return 1;
  }
};

struct Zero {
  using expression_tag = void;

  template <Numeric... Values>
  constexpr int evaluate(const Values &...) const {
    return 0;
  }
};

template <std::size_t Index>
struct Variable {
  using expression_tag = void;

  template <Numeric... Values>
    requires(Index < sizeof...(Values))
  constexpr auto evaluate(const Values &...values) const {
    return std::get<Index>(std::tie(values...));
  }
};

constexpr Variable<0> x{};
constexpr Variable<1> y{};

template <typename Variable>
struct is_variable : std::false_type {};

template <std::size_t Index>
struct is_variable<Variable<Index>> : std::true_type {};

template <typename Variable>
inline constexpr bool is_variable_v =
    is_variable<std::remove_cvref_t<Variable>>::value;

template <typename T>
concept VariableNode = is_variable_v<T> && is_expression_v<T>;

template <Expression E, Numeric... Values>
constexpr auto evaluate(const E &expression, const Values &...values)
    -> decltype(expression.evaluate(values...)) {
  return expression.evaluate(values...);
}

template <ArithmeticValue V>
constexpr auto constant(V value) {
  return Constant{value};
}

} // namespace expr_ad
