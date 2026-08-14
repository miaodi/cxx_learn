#pragma once
#include "meta/integral_constant.hpp"
#include <cstddef>
namespace meta {

// Milestone 3: is_array, remove_extent, remove_all_extents, rank, extent,
// and their helpers
template <typename T>
struct is_array : false_type {};

template <typename T>
struct is_array<T[]> : true_type {};

template <typename T, std::size_t N>
struct is_array<T[N]> : true_type {};

template <typename T>
inline constexpr bool is_array_v = is_array<T>::value;
} // namespace meta
