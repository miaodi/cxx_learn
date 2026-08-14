#include "expr_ad/core.hpp"

#include <concepts>
#include <type_traits>

namespace {

using namespace expr_ad;

struct NotAnExpression {};

static_assert(Expression<Constant<double>>);
static_assert(Expression<const Variable<0> &>);
static_assert(!Expression<int>);
static_assert(!Expression<NotAnExpression>);
static_assert(is_expression_v<Multiply<Variable<0>, Constant<int>>>);

constexpr Constant<double> three_halves{1.5};
static_assert(three_halves.evaluate() == 1.5);
static_assert(three_halves.evaluate(7, 8) == 1.5);

constexpr Variable<0> first{};
constexpr Variable<1> second{};
static_assert(first.evaluate(2.0, 5.0) == 2.0);
static_assert(second.evaluate(2.0, 5.0) == 5.0);

constexpr Add<Variable<0>, Constant<int>> shifted{first, {3}};
static_assert(shifted.evaluate(4) == 7);
static_assert(evaluate(shifted, 4) == 7);

constexpr Multiply<decltype(shifted), Variable<1>> product{shifted, second};
static_assert(product.evaluate(4, 2) == 14);

static_assert(Zero{}.evaluate(99) == 0);
static_assert(One{}.evaluate(99) == 1);

} // namespace

int main() {}
