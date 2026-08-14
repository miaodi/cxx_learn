#pragma once

namespace meta {

// Milestone 1: integral_constant, bool_constant, true_type, false_type
template <typename T, T v>
struct integral_constant {
  using type = integral_constant<T, v>;
  using value_type = T;
  static constexpr T value = v;

  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <bool V>
struct bool_constant : integral_constant<bool, V> {};

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;
} // namespace meta
