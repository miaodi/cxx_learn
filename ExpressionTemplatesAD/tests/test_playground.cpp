#include "expr_ad/expr_ad.hpp"

#include <complex>
#include <concepts>
#include <iostream>
#include <string>
namespace {

using namespace expr_ad;

// Keep this target as a scratchpad throughout the course. Add compile-time
// examples for the current milestone and replace assertions as the design
// grows.

template <typename ConstantType, typename... Values>
concept CanEvaluateWith =
    requires(const ConstantType &constant, Values &&...values) {
      constant.evaluate(static_cast<Values &&>(values)...);
    };

static_assert(ArithmeticValue<int>);
static_assert(ArithmeticValue<double>);
static_assert(!ArithmeticValue<std::string>);

static_assert(Numeric<int>);
static_assert(Numeric<double>);
static_assert(!Numeric<std::string>);
static_assert(!ArithmeticValue<std::complex<double>>);
static_assert(Numeric<std::complex<double>>);

constexpr Constant<double> three_halves{1.5};
static_assert(is_expression_v<decltype(three_halves)>);
static_assert(
    std::same_as<typename decltype(three_halves)::value_type, double>);
static_assert(three_halves.evaluate() == 1.5);
static_assert(three_halves.evaluate(1, 2.0, 3.0f) == 1.5);

static_assert(CanEvaluateWith<Constant<double>, int, double, float>);

constexpr Zero zero;
static_assert(zero.evaluate() == 0);
} // namespace

int main() {
  constexpr Zero zero;
  zero.evaluate();
  zero.evaluate(1, 2.0, 3.0f);
  int x_val = 0;
  zero.evaluate(x_val);
  Variable<0> var0;
  std::cout << var0.evaluate(1, 2.0, 3.0f) << std::endl;
  int y_val = 42;
  auto result = var0.evaluate(y_val);
  static_assert(std::same_as<decltype(result), int>);
  std::cout << var0.evaluate(y_val) << std::endl;

  Add<Variable<0>, Constant<double>> expr{var0, three_halves};
  std::cout << expr.evaluate(1, 2.0, 3.0f) << std::endl;
  std::cout << expr.evaluate(y_val) << std::endl;

  auto expr2 = sin(var0);
  auto expr3 = cos(expr2);
  std::cout << expr2.evaluate(1, 2.0, 3.0f) << std::endl;
  std::cout << expr3.evaluate(1, 2.0, 3.0f) << std::endl;
  auto expr4 = -expr3 + (three_halves);
  std::cout << expr4.evaluate(1, 2.0, 3.0f) << std::endl;

  constexpr auto expr5 = exp(x) * sin(cos(y));

  std::cout << expr5.evaluate(x_val, y_val) << std::endl;

  std::cout << differentiate<0>(expr5).evaluate(x_val, y_val) << std::endl;
  std::cout << differentiate<1>(expr5).evaluate(x_val, y_val) << std::endl;
  constexpr auto expr6 = simplify(differentiate<0>(expr5));
  std::cout << expr6.evaluate(x_val, y_val) << std::endl;
  return 0;
}
