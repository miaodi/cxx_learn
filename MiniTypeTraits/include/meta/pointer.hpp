#pragma once
#include "meta/integral_constant.hpp"
#include "meta/reference.hpp"
namespace meta {

// Milestone 2: remove_pointer, is_pointer, add_pointer, and their helpers
template <typename T>
struct remove_pointer {
  using type = T;
};

template <typename T>
struct remove_pointer<T *> {
  using type = T;
};

template <typename T>
struct remove_pointer<T * const> {
  using type = T;
};

template <typename T>
struct remove_pointer<T * volatile> {
  using type = T;
};

template <typename T>
struct remove_pointer<T * const volatile> {
  using type = T;
};

template <typename T>
using remove_pointer_t = typename remove_pointer<T>::type;

template <typename T>
struct add_pointer {
  using type = remove_reference_t<T> *;
};

template <typename T>
using add_pointer_t = add_pointer<T>::type;

// predicates
template <typename T>
struct is_pointer : false_type {};

template <typename T>
struct is_pointer<T *> : true_type {};

template <typename T>
struct is_pointer<T *const> : true_type {};
template <typename T>
struct is_pointer<T *volatile> : true_type {};
template <typename T>
struct is_pointer<T *const volatile> : true_type {};

template <typename T>
constexpr bool is_pointer_v = is_pointer<T>::value;


} // namespace meta
