#pragma once
#include "meta/integral_constant.hpp"
namespace meta {

// Milestone 4: void_t, has_value_type, has_begin, is_addable
template <typename...>
using void_t = void;

template <typename T, typename = void>
struct has_value_type : false_type {};

template <typename T>
struct has_value_type<T, void_t<typename T::value_type>> : true_type {};

template <typename T>
inline constexpr bool has_value_type_v = has_value_type<T>::value;

template <typename T>
concept HasBegin = requires(T &t) { t.begin(); };
template <typename T>
struct has_begin : bool_constant<HasBegin<T>> {};

template <typename T>
inline constexpr bool has_begin_v = has_begin<T>::value;

template <typename T, typename U>
concept Addable = requires(T t, U u) { t + u; };

template <typename T, typename U = T>
struct is_addable : bool_constant<Addable<T, U>> {};

template <typename T, typename U = T>
inline constexpr bool is_addable_v = is_addable<T, U>::value;
} // namespace meta
