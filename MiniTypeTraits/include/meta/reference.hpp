#pragma once
#include "meta/integral_constant.hpp"
#include <type_traits>
namespace meta {

// Milestones 1-2: remove_reference, reference predicates,
// add_lvalue_reference, add_rvalue_reference, and their helpers
template <typename T>
struct remove_reference {
  using type = T;
};

template <typename T>
struct remove_reference<T &> {
  using type = T;
};

// type&& is rvalue reference
template <typename T>
struct remove_reference<T &&> {
  using type = T;
};

template <typename T>
using remove_reference_t = remove_reference<T>::type;

template <typename T, typename = void>
struct add_lvalue_reference {
  using type = T;
};

template <typename T>
struct add_lvalue_reference<T, std::void_t<T &>> {
  using type = T &;
};

template <typename T>
using add_lvalue_reference_t = add_lvalue_reference<T>::type;

template <typename T, typename = void>
struct add_rvalue_reference {
  using type = T;
};

template <typename T>
struct add_rvalue_reference<T, std::void_t<T &&>> {
  using type = T &&;
};

template <typename T>
using add_rvalue_reference_t = add_rvalue_reference<T>::type;

// predicates
template <typename T>
struct is_reference : false_type {};

template <typename T>
struct is_reference<T &> : true_type {};

template <typename T>
struct is_reference<T &&> : true_type {};

template <typename T>
constexpr bool is_reference_v = is_reference<T>::value;

template <typename T>
struct is_lvalue_reference : false_type {};

template <typename T>
struct is_lvalue_reference<T &> : true_type {};

template <typename T>
constexpr bool is_lvalue_reference_v = is_lvalue_reference<T>::value;

template <typename T>
struct is_rvalue_reference : false_type {};

template <typename T>
struct is_rvalue_reference<T &&> : true_type {};

template <typename T>
constexpr bool is_rvalue_reference_v = is_rvalue_reference<T>::value;
} // namespace meta
