#pragma once
#include "meta/integral_constant.hpp"
namespace meta {

// Milestones 1-2: remove_const, remove_volatile, remove_cv,
// is_const, is_volatile, and their _t/_v helpers
template <typename T>
struct remove_const {
  using type = T;
};

template <typename T>
struct remove_const<const T> {
  using type = T;
};

template <typename T>
using remove_const_t = typename remove_const<T>::type;

template <typename T>
struct remove_volatile {
  using type = T;
};

template <typename T>
struct remove_volatile<volatile T> {
  using type = T;
};

template <typename T>
using remove_volatile_t = typename remove_volatile<T>::type;

template <typename T>
struct remove_cv {
  using type = T;
};

template <typename T>
struct remove_cv<volatile T> {
  using type = T;
};
template <typename T>
struct remove_cv<const T> {
  using type = T;
};

template <typename T>
struct remove_cv<const volatile T> {
  using type = T;
};
template <typename T>
using remove_cv_t = remove_cv<T>::type;

template <typename T>
struct is_const : false_type {};

template <typename T>
struct is_const<const T> : true_type {};

template <typename T>
constexpr bool is_const_v = is_const<T>::value;

template <typename T>
struct is_volatile : false_type {};

template <typename T>
struct is_volatile<volatile T> : true_type {};

template <typename T>
constexpr bool is_volatile_v = is_volatile<T>::value;
} // namespace meta
