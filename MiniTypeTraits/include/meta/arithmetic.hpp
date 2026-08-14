#pragma once
#include "meta/comparison.hpp"
#include "meta/cv.hpp"
#include "meta/integral_constant.hpp"
namespace meta {

// Milestone 2: is_void, is_integral, is_floating_point, is_arithmetic,
// and their _v helpers

template <typename T>
struct is_void : is_same<remove_cv_t<T>, void> {};

template <typename T>
inline constexpr bool is_void_v = is_void<T>::value;

template <typename T>
struct is_integral
    : bool_constant<is_any_of_v<
          remove_cv_t<T>, bool, char, signed char, unsigned char, wchar_t,
          char8_t, char16_t, char32_t, short, unsigned short, int, unsigned int,
          long, unsigned long, long long, unsigned long long>> {};

template <typename T>
inline constexpr bool is_integral_v = is_integral<T>::value;

template <typename T>
struct is_floating_point
    : bool_constant<is_any_of_v<remove_cv_t<T>, float, double, long double>> {};

template <typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

template <typename T>
struct is_arithmetic
    : bool_constant<is_floating_point_v<T> || is_integral_v<T>> {};

template <typename T>
inline constexpr bool is_arithmetic_v = is_arithmetic<T>::value;
} // namespace meta
