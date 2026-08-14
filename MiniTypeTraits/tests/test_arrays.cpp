#include "meta/type_traits.hpp"

#include <cstddef>
#include <type_traits>

using matrix = const int[2][3];
using unbounded = int[][3];

static_assert(meta::is_array_v<int> == std::is_array_v<int>);
static_assert(meta::is_array_v<matrix> == std::is_array_v<matrix>);
static_assert(meta::is_array_v<unbounded> == std::is_array_v<unbounded>);
static_assert(meta::is_same_v<meta::remove_extent_t<matrix>,
                              std::remove_extent_t<matrix>>);
static_assert(meta::is_same_v<meta::remove_all_extents_t<matrix>,
                              std::remove_all_extents_t<matrix>>);
static_assert(meta::rank_v<matrix> == std::rank_v<matrix>);
static_assert(meta::extent_v<matrix, 0> == std::extent_v<matrix, 0>);
static_assert(meta::extent_v<matrix, 1> == std::extent_v<matrix, 1>);
static_assert(meta::extent_v<matrix, 2> == std::extent_v<matrix, 2>);

int main() {}
