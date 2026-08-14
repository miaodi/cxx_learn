#pragma once

namespace my_type_traits {
template <typename T, T V>
struct integral_constant {
  static constexpr T value = V;
  using value_type = T;
  using type = integral_constant<T, V>;
  constexpr operator value_type() const noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <bool V>
struct bool_constant : integral_constant<bool, V> {};

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <typename T>
struct remove_reference {
  using type = T;
};

template <typename T>
struct remove_reference<T &> {
  using type = T;
};

template <typename T>
struct remove_reference<T &&> {
  using type = T;
};
template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

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
struct remove_pointer {
  using type = T;
};

template <typename T>
struct remove_pointer<T *> {
  using type = T;
};

template <typename T>
using remove_pointer_t = typename remove_pointer<T>::type;

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
  using type = remove_volatile_t<remove_const_t<T>>;
};

template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

template <typename T, typename U>
struct is_same : false_type {};

template <typename T>
struct is_same<T, T> : true_type {};

template <typename T, typename U>
constexpr bool is_same_v = is_same<T, U>::value;

template <typename T>
struct is_const : false_type {};

template <typename T>
struct is_const<const T> : true_type {};

template <typename T>
constexpr bool is_const_v = is_const<T>::value;

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

template <typename T>
struct is_pointer : false_type {};

template <typename T>
struct is_pointer<T *> : true_type {};

template <typename T>
constexpr bool is_pointer_v = is_pointer<T>::value;
} // namespace my_type_traits