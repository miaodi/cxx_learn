#pragma once

#include "expr_ad/core.hpp"
#include <cmath>

namespace expr_ad {

// Milestone 3: implement the unary expression nodes Negate, Sine, Cosine,
// Exponential, and Logarithm, then expose constrained unary -, sin, cos, exp,
// and log builders.
template <Expression E>
struct Negate {
  using expression_tag = void;

  E expr;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return -expr.evaluate(values...);
  }
};

template <Expression E>
struct Sine {
  using expression_tag = void;

  E expr;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return std::sin(expr.evaluate(values...));
  }
};
template <Expression E>
struct Cosine {
  using expression_tag = void;

  E expr;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return std::cos(expr.evaluate(values...));
  }
};

template <Expression E>
struct Exponential {
  using expression_tag = void;

  E expr;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return std::exp(expr.evaluate(values...));
  }
};

template <Expression E>
struct Logarithm {
  using expression_tag = void;

  E expr;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return std::log(expr.evaluate(values...));
  }
};

template <Expression E>
constexpr auto operator-(E &&expr) {
  return Negate{std::forward<E>(expr)};
}

template <Expression E>
constexpr auto sin(E &&expr) {
  return Sine{std::forward<E>(expr)};
}

template <Expression E>
constexpr auto cos(E &&expr) {
  return Cosine{std::forward<E>(expr)};
}

template <Expression E>
constexpr auto exp(E &&expr) {
  return Exponential{std::forward<E>(expr)};
}

template <Expression E>
constexpr auto log(E &&expr) {
  return Logarithm{std::forward<E>(expr)};
}
} // namespace expr_ad
