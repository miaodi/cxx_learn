#include "expr_ad/binary.hpp"

#include <concepts>
#include <type_traits>

namespace {

using namespace expr_ad;

struct Unsupported {};

static_assert(is_expression_v<Multiply<Variable<0>, Constant<int>>>);

constexpr Variable<0> first{};
constexpr Variable<1> second{};
constexpr Add<Variable<0>, Constant<int>> shifted{first, {3}};
static_assert(shifted.evaluate(4) == 7);
static_assert(evaluate(shifted, 4) == 7);

constexpr Multiply<decltype(shifted), Variable<1>> product{shifted, second};
static_assert(product.evaluate(4, 2) == 14);

constexpr auto answer = constant(42);
static_assert(std::same_as<std::remove_cvref_t<decltype(answer)>,
                           Constant<int>>);
static_assert(evaluate(answer) == 42);

constexpr auto square_plus_two = x * x + 2.0;
using Expected = Add<Multiply<Variable<0>, Variable<0>>, Constant<double>>;

static_assert(
    std::same_as<std::remove_cvref_t<decltype(square_plus_two)>, Expected>);
static_assert(square_plus_two.evaluate(3.0) == 11.0);
static_assert(evaluate(square_plus_two, 4.0) == 18.0);

constexpr auto scalar_on_left = 2 * x + 1;
static_assert(evaluate(scalar_on_left, 5) == 11);

constexpr auto difference = x - 2.0;
using ExpectedDifference = Subtract<Variable<0>, Constant<double>>;
static_assert(std::same_as<std::remove_cvref_t<decltype(difference)>,
                           ExpectedDifference>);
static_assert(evaluate(difference, 5.0) == 3.0);

constexpr auto quotient = 12.0 / (x + 1.0);
static_assert(evaluate(quotient, 3.0) == 3.0);

constexpr auto make_owned_expression() {
  auto local = x + 2.5;
  return local * local;
}

constexpr auto owned = make_owned_expression();
static_assert(evaluate(owned, 1.5) == 16.0);

static_assert(is_expression_v<decltype(owned)>);
static_assert(ExpressionOperand<Variable<0>>);
static_assert(ExpressionOperand<double>);
static_assert(!ExpressionOperand<Unsupported>);

// The constrained overloads must not replace ordinary scalar arithmetic.
static_assert(std::same_as<decltype(1 + 2), int>);
static_assert(std::same_as<decltype(4 - 2), int>);
static_assert(std::same_as<decltype(2.0 * 3.0), double>);
static_assert(std::same_as<decltype(6.0 / 2.0), double>);

} // namespace

int main() {}
