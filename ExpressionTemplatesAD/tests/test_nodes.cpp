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

constexpr Constant<double> three_halves{1.5};
static_assert(three_halves.evaluate() == 1.5);
static_assert(three_halves.evaluate(7, 8) == 1.5);

constexpr Variable<0> first{};
constexpr Variable<1> second{};
static_assert(first.evaluate(2.0, 5.0) == 2.0);
static_assert(second.evaluate(2.0, 5.0) == 5.0);
static_assert(std::same_as<std::remove_cvref_t<decltype(x)>, Variable<0>>);
static_assert(std::same_as<std::remove_cvref_t<decltype(y)>, Variable<1>>);
static_assert(x.evaluate(2.0, 5.0) == 2.0);
static_assert(y.evaluate(2.0, 5.0) == 5.0);

static_assert(Zero{}.evaluate(99) == 0);
static_assert(One{}.evaluate(99) == 1);

} // namespace

int main() {}
