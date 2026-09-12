#pragma once

#include "expr_ad/core.hpp"

#include <cmath>
#include <type_traits>
#include <utility>

namespace expr_ad {

template <Expression Operand>
struct Negate {
  using expression_tag = void;

  Operand operand;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    return -operand.evaluate(values...);
  }
};

template <Expression Operand>
struct Sine {
  using expression_tag = void;

  Operand operand;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    using std::sin;
    return sin(operand.evaluate(values...));
  }
};

template <Expression Operand>
struct Cosine {
  using expression_tag = void;

  Operand operand;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    using std::cos;
    return cos(operand.evaluate(values...));
  }
};

template <Expression Operand>
struct Exponential {
  using expression_tag = void;

  Operand operand;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    using std::exp;
    return exp(operand.evaluate(values...));
  }
};

template <Expression Operand>
struct Logarithm {
  using expression_tag = void;

  Operand operand;

  template <Numeric... Values>
  constexpr auto evaluate(const Values &...values) const {
    using std::log;
    return log(operand.evaluate(values...));
  }
};

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

template <Expression E>
constexpr auto log(E &&expression) {
  using Stored = std::remove_cvref_t<E>;
  return Logarithm<Stored>{std::forward<E>(expression)};
}

} // namespace expr_ad
