#include "expr_ad/expr_ad.hpp"

#include <cmath>

namespace {

bool near(double left, double right, double tolerance = 1.0e-9) {
  return std::abs(left - right) <= tolerance;
}

} // namespace

int main() {
  using namespace expr_ad;

  constexpr Dual<double> seeded{3.0, 1.0};
  constexpr auto square = seeded * seeded;
  static_assert(square.value == 9.0);
  static_assert(square.derivative == 6.0);
  constexpr auto halved = seeded / Dual<double>{2.0};
  static_assert(halved.value == 1.5);
  static_assert(halved.derivative == 0.5);

  const auto expression = 2.0 + exp(x) * sin(x) + x * x;
  constexpr double point = 0.4;

  const auto dual_result = evaluate(expression, Dual<double>{point, 1.0});
  const double expected_value =
      2.0 + std::exp(point) * std::sin(point) + point * point;
  const double expected_derivative =
      std::exp(point) * (std::sin(point) + std::cos(point)) + 2.0 * point;

  if (!near(dual_result.value, expected_value) ||
      !near(dual_result.derivative, expected_derivative)) {
    return 1;
  }

  const auto symbolic = simplify(differentiate<0>(expression));
  if (!near(evaluate(symbolic, point), expected_derivative)) {
    return 1;
  }

  constexpr double step = 1.0e-6;
  const double finite_difference = (evaluate(expression, point + step) -
                                    evaluate(expression, point - step)) /
                                   (2.0 * step);
  if (!near(finite_difference, expected_derivative, 1.0e-6)) {
    return 1;
  }

  const auto two_variable = x * y + sin(y);
  const auto dx =
      evaluate(two_variable, Dual<double>{2.0, 1.0}, Dual<double>{0.5, 0.0});
  const auto dy =
      evaluate(two_variable, Dual<double>{2.0, 0.0}, Dual<double>{0.5, 1.0});
  if (!near(dx.derivative, 0.5) || !near(dy.derivative, 2.0 + std::cos(0.5))) {
    return 1;
  }

  return 0;
}
