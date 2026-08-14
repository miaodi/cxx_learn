#include "meta/type_traits.hpp"

#include <type_traits>

static_assert(meta::is_same_v<meta::conditional_t<true, int, double>, int>);
static_assert(meta::is_same_v<meta::conditional_t<false, int, double>, double>);
static_assert(meta::is_same_v<meta::enable_if_t<true, long>, long>);

static_assert(meta::is_integral_v<const int> == std::is_integral_v<const int>);
static_assert(meta::is_integral_v<int &> == std::is_integral_v<int &>);
static_assert(meta::is_integral_v<wchar_t> == std::is_integral_v<wchar_t>);
static_assert(meta::is_floating_point_v<volatile double> ==
              std::is_floating_point_v<volatile double>);
static_assert(meta::is_floating_point_v<int> ==
              std::is_floating_point_v<int>);
static_assert(meta::is_arithmetic_v<unsigned long long> ==
              std::is_arithmetic_v<unsigned long long>);
static_assert(meta::is_arithmetic_v<void> == std::is_arithmetic_v<void>);

int main() {}
