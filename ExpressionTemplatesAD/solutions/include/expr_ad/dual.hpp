#pragma once

#include <cmath>

namespace expr_ad {

template <typename T>
struct Dual {
  T value{};
  T derivative{};

  constexpr Dual() = default;
  constexpr Dual(T value_in, T derivative_in = {})
      : value(value_in), derivative(derivative_in) {}

  friend constexpr Dual operator+(Dual left, Dual right) {
    return {left.value + right.value, left.derivative + right.derivative};
  }

  friend constexpr Dual operator-(Dual left, Dual right) {
    return {left.value - right.value, left.derivative - right.derivative};
  }

  friend constexpr Dual operator*(Dual left, Dual right) {
    return {left.value * right.value,
            left.derivative * right.value + left.value * right.derivative};
  }

  friend constexpr Dual operator/(Dual left, Dual right) {
    return {left.value / right.value,
            (left.derivative * right.value - left.value * right.derivative) /
                (right.value * right.value)};
  }

  friend constexpr Dual operator-(Dual input) {
    return {-input.value, -input.derivative};
  }

  friend constexpr bool operator==(const Dual &, const Dual &) = default;
};

template <typename T>
constexpr Dual<T> sin(Dual<T> input) {
  using std::cos;
  using std::sin;
  return {sin(input.value), cos(input.value) * input.derivative};
}

template <typename T>
constexpr Dual<T> cos(Dual<T> input) {
  using std::cos;
  using std::sin;
  return {cos(input.value), -sin(input.value) * input.derivative};
}

template <typename T>
constexpr Dual<T> exp(Dual<T> input) {
  using std::exp;
  const auto exponential = exp(input.value);
  return {exponential, exponential * input.derivative};
}

} // namespace expr_ad
