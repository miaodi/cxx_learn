#pragma once

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace expr_ad {

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
concept Numeric =
    std::copy_constructible<std::remove_cvref_t<T>> &&
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
  using expression_tag = void;
  using value_type = T;

  T value;

  template <Numeric... Values>
  constexpr T evaluate(const Values &...) const {
    return value;
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

inline constexpr Variable<0> x{};
inline constexpr Variable<1> y{};

struct Zero {
  using expression_tag = void;

  template <Numeric... Values>
  constexpr int evaluate(const Values &...) const {
    return 0;
  }
};

struct One {
  using expression_tag = void;

  template <Numeric... Values>
  constexpr int evaluate(const Values &...) const {
    return 1;
  }
};

template <Expression E, Numeric... Values>
constexpr auto evaluate(const E &expression, const Values &...values)
    -> decltype(expression.evaluate(values...)) {
  return expression.evaluate(values...);
}

} // namespace expr_ad
