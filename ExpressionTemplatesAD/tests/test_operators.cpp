#include "expr_ad/operators.hpp"

#include <concepts>
#include <type_traits>

namespace {

using namespace expr_ad;

struct Unsupported {};

constexpr auto square_plus_two = x * x + 2.0;
using Expected = Add<Multiply<Variable<0>, Variable<0>>, Constant<double>>;

static_assert(
    std::same_as<std::remove_cvref_t<decltype(square_plus_two)>, Expected>);
static_assert(square_plus_two.evaluate(3.0) == 11.0);
static_assert(evaluate(square_plus_two, 4.0) == 18.0);

constexpr auto scalar_on_left = 2 * x + 1;
static_assert(evaluate(scalar_on_left, 5) == 11);

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
static_assert(std::same_as<decltype(2.0 * 3.0), double>);

} // namespace

int main() {}
