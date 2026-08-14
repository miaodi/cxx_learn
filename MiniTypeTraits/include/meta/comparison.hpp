#pragma once
#include "meta/integral_constant.hpp"

namespace meta {

// Milestone 1: is_same, is_same_v
template <typename U, typename V>
struct is_same : false_type {};

template <typename U>
struct is_same<U, U> : true_type {};

template <typename U, typename V>
constexpr bool is_same_v = is_same<U, V>::value;

template <typename T, typename... Args>
constexpr bool is_any_of_v =
    (is_same_v<T, Args> || ...); // C++17 fold expression TODO: needs review
} // namespace meta
