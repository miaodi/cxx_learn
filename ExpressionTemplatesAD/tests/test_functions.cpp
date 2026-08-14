#include "expr_ad/functions.hpp"

#include <cmath>

namespace {

bool near(double left, double right, double tolerance = 1.0e-12) {
  return std::abs(left - right) <= tolerance;
}

} // namespace

int main() {
  using namespace expr_ad;

  const auto expression = exp(x) / (1.0 + sin(-y));
  constexpr double x_value = 0.4;
  constexpr double y_value = 0.2;
  const double expected = std::exp(x_value) / (1.0 + std::sin(-y_value));
  if (!near(evaluate(expression, x_value, y_value), expected)) {
    return 1;
  }

  const auto all_operations = cos(x) - sin(y) + (-x) / 2.0;
  const double all_expected =
      std::cos(x_value) - std::sin(y_value) - x_value / 2.0;
  if (!near(evaluate(all_operations, x_value, y_value), all_expected)) {
    return 1;
  }

  const auto reciprocal = 1.0 / (x - 1.0);
  constexpr double near_singular_point = 1.000001;
  const double reciprocal_expected = 1.0 / (near_singular_point - 1.0);
  if (!near(evaluate(reciprocal, near_singular_point), reciprocal_expected)) {
    return 1;
  }

  return 0;
}
