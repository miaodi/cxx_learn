#include "expr_ad/differentiate.hpp"

#include <cmath>
#include <concepts>
#include <type_traits>

namespace {

bool near(double left, double right, double tolerance = 1.0e-11) {
  return std::abs(left - right) <= tolerance;
}

using namespace expr_ad;

static_assert(std::same_as<derivative_t<decltype(x), decltype(x)>, One>);
static_assert(std::same_as<derivative_t<decltype(y), decltype(x)>, Zero>);
static_assert(std::same_as<derivative_t<Constant<double>, Variable<0>>, Zero>);

constexpr auto simple_expression = x * x + sin(x);
using RawSimpleDerivative =
    Add<Add<Multiply<One, Variable<0>>, Multiply<Variable<0>, One>>,
        Multiply<Cosine<Variable<0>>, One>>;
static_assert(
    std::same_as<
        std::remove_cvref_t<decltype(differentiate<0>(simple_expression))>,
        RawSimpleDerivative>);

} // namespace

int main() {
  using namespace expr_ad;

  constexpr double x_value = 0.7;
  constexpr double y_value = -0.3;

  const auto expression = x * x + sin(x) + exp(y) / (x + 2.0);
  const auto with_respect_to_x = differentiate<0>(expression);
  const auto with_respect_to_y = differentiate<1>(expression);

  const double expected_dx =
      2.0 * x_value + std::cos(x_value) -
      std::exp(y_value) / ((x_value + 2.0) * (x_value + 2.0));
  const double expected_dy = std::exp(y_value) / (x_value + 2.0);

  if (!near(evaluate(with_respect_to_x, x_value, y_value), expected_dx)) {
    return 1;
  }
  if (!near(evaluate(with_respect_to_y, x_value, y_value), expected_dy)) {
    return 1;
  }

  const auto chained = differentiate<0>(sin(x * x));
  const double expected_chain = std::cos(x_value * x_value) * 2.0 * x_value;
  if (!near(evaluate(chained, x_value), expected_chain)) {
    return 1;
  }

  const auto cosine_of_negative = differentiate<0>(cos(-x));
  if (!near(evaluate(cosine_of_negative, x_value), -std::sin(x_value))) {
    return 1;
  }

  const auto difference = differentiate<0>(x - y);
  if (!near(evaluate(difference, x_value, y_value), 1.0)) {
    return 1;
  }

  return 0;
}
