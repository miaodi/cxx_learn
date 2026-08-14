#include "expr_ad/simplify.hpp"

#include <cmath>
#include <concepts>
#include <type_traits>

namespace {

using namespace expr_ad;

using Noisy = Multiply<One, Add<Zero, Variable<0>>>;
static_assert(std::same_as<simplified_t<Noisy>, Variable<0>>);

using DoubleNegative = Negate<Negate<Variable<0>>>;
static_assert(std::same_as<simplified_t<DoubleNegative>, Variable<0>>);

using Annihilated = Multiply<Add<Variable<0>, One>, Zero>;
static_assert(std::same_as<simplified_t<Annihilated>, Zero>);

using SubtractZero = Subtract<Variable<0>, Zero>;
static_assert(std::same_as<simplified_t<SubtractZero>, Variable<0>>);

using DivideOne = Divide<Variable<1>, One>;
static_assert(std::same_as<simplified_t<DivideOne>, Variable<1>>);

constexpr auto product_expression = x * y;
constexpr auto product_dx = simplify(differentiate<0>(product_expression));
static_assert(
    std::same_as<std::remove_cvref_t<decltype(product_dx)>, Variable<1>>);
static_assert(evaluate(product_dx, 3.0, 4.0) == 4.0);

constexpr auto simple_expression = x * x + sin(x);
using CompactDerivative =
    Add<Add<Variable<0>, Variable<0>>, Cosine<Variable<0>>>;
static_assert(std::same_as<std::remove_cvref_t<decltype(simplify(
                               differentiate<0>(simple_expression)))>,
                           CompactDerivative>);

} // namespace

int main() {
  using namespace expr_ad;

  constexpr double point = 0.6;
  const auto derivative = simplify(differentiate<0>(simple_expression));
  return std::abs(evaluate(derivative, point) -
                  (2.0 * point + std::cos(point))) < 1.0e-12
             ? 0
             : 1;
}
