#pragma once
#include "meta/conditional.hpp"
#include "meta/integral_constant.hpp"

namespace meta {

// Milestone 3: conjunction, disjunction, negation, and their _v helpers.
//
// Each operand is a trait type with a static Boolean `value`, such as
// true_type, false_type, or is_integral<int>.
//
// conjunction<B...> is the type-level equivalent of B1 && B2 && ...:
// - an empty pack is true;
// - otherwise, inherit from the first false operand, or recurse past a true
//   operand. This makes later operands remain uninstantiated after false.

template <typename...>
struct conjunction;

template <>
struct conjunction<> : true_type {};

template <typename T>
struct conjunction<T> : T {};

template <typename Arg1, typename... Args>
struct conjunction<Arg1, Args...>
    : conditional_t<Arg1::value, conjunction<Args...>, Arg1> {};

template <typename... Args>
inline constexpr bool conjunction_v = conjunction<Args...>::value;

// disjunction<B...> similarly models B1 || B2 || ...:
// - an empty pack is false;
// - inherit from the first true operand, or recurse past a false operand.

template <typename...>
struct disjunction;

template <>
struct disjunction<> : false_type {};

template <typename T>
struct disjunction<T> : T {};

template <typename Arg1, typename... Args>
struct disjunction<Arg1, Args...>
    : conditional_t<Arg1::value, Arg1, disjunction<Args...>> {};

template <typename... Args>
inline constexpr bool disjunction_v = disjunction<Args...>::value;

// negation<B> simply wraps !B::value in bool_constant.

template <typename T>
struct negation : conditional_t<T::value, false_type, true_type> {};

template <typename T>
inline constexpr bool negation_v = negation<T>::value;

// Hint: partial specializations plus conditional_t can select the next type.
// A fold such as (B::value && ...) short-circuits value evaluation, but the
// compiler must still form and type-check every B::value expression. That can
// instantiate an invalid later trait such as requires_complete<incomplete>.
// Recursive type selection avoids instantiating the remaining traits once the
// result is already known.

} // namespace meta
