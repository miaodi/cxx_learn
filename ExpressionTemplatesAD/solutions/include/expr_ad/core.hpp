#pragma once

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
struct Constant {
  using expression_tag = void;
  using value_type = T;

  T value;

  template <typename... Values>
  constexpr T evaluate(const Values &...) const {
    return value;
  }
};

template <std::size_t Index>
struct Variable {
  using expression_tag = void;

  template <typename... Values>
    requires(Index < sizeof...(Values))
  constexpr auto evaluate(const Values &...values) const {
    return std::get<Index>(std::tie(values...));
  }
};

struct Zero {
  using expression_tag = void;

  template <typename... Values>
  constexpr int evaluate(const Values &...) const {
    return 0;
  }
};

struct One {
  using expression_tag = void;

  template <typename... Values>
  constexpr int evaluate(const Values &...) const {
    return 1;
  }
};

template <Expression Left, Expression Right>
struct Add {
  using expression_tag = void;

  Left left;
  Right right;

  template <typename... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) + right.evaluate(values...);
  }
};

template <Expression Left, Expression Right>
struct Multiply {
  using expression_tag = void;

  Left left;
  Right right;

  template <typename... Values>
  constexpr auto evaluate(const Values &...values) const {
    return left.evaluate(values...) * right.evaluate(values...);
  }
};

template <Expression E, typename... Values>
constexpr auto evaluate(const E &expression, const Values &...values)
    -> decltype(expression.evaluate(values...)) {
  return expression.evaluate(values...);
}

} // namespace expr_ad
