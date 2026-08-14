#include "meta/type_traits.hpp"

#include <type_traits>

static_assert(meta::integral_constant<int, 42>::value == 42);
static_assert(meta::integral_constant<int, 42>{}() == 42);
static_assert(meta::true_type::value);
static_assert(!meta::false_type::value);

static_assert(meta::is_same_v<int, int> == std::is_same_v<int, int>);
static_assert(meta::is_same_v<int, const int> == std::is_same_v<int, const int>);

static_assert(meta::is_same_v<meta::remove_const_t<const int>,
                              std::remove_const_t<const int>>);
static_assert(meta::is_same_v<meta::remove_reference_t<int &>,
                              std::remove_reference_t<int &>>);
static_assert(meta::is_same_v<meta::remove_reference_t<int &&>,
                              std::remove_reference_t<int &&>>);

static_assert(meta::is_const_v<const int> == std::is_const_v<const int>);
static_assert(meta::is_const_v<const int &> == std::is_const_v<const int &>);
static_assert(meta::is_lvalue_reference_v<int &> ==
              std::is_lvalue_reference_v<int &>);
static_assert(meta::is_rvalue_reference_v<int &&> ==
              std::is_rvalue_reference_v<int &&>);
static_assert(meta::is_reference_v<int> == std::is_reference_v<int>);
static_assert(meta::is_reference_v<int &> == std::is_reference_v<int &>);
static_assert(meta::is_reference_v<int &&> == std::is_reference_v<int &&>);

int main() {}
