#include "expr_ad/binary.hpp"
#include "expr_ad/unary.hpp"

#include <cmath>
#include <concepts>
#include <type_traits>

namespace {

bool near(double left, double right, double tolerance = 1.0e-12) {
  return std::abs(left - right) <= tolerance;
}

using namespace expr_ad;

constexpr auto negative_x = -x;
static_assert(std::same_as<std::remove_cvref_t<decltype(negative_x)>,
                           Negate<Variable<0>>>);

constexpr auto logarithm_x = log(x);
static_assert(std::same_as<std::remove_cvref_t<decltype(logarithm_x)>,
                           Logarithm<Variable<0>>>);

} // namespace

int main() {
  using namespace expr_ad;

  constexpr double x_value = 0.4;
  constexpr double y_value = 0.2;

  const auto expression = exp(x) * (1.0 + sin(-y));
  const double expected =
      std::exp(x_value) * (1.0 + std::sin(-y_value));
  if (!near(evaluate(expression, x_value, y_value), expected)) {
    return 1;
  }

  const auto all_unary = cos(x) + sin(y) + (-x) * 2.0 + log(x + 1.0);
  const double all_expected =
      std::cos(x_value) + std::sin(y_value) - x_value * 2.0 +
      std::log(x_value + 1.0);
  if (!near(evaluate(all_unary, x_value, y_value), all_expected)) {
    return 1;
  }

  return 0;
}
