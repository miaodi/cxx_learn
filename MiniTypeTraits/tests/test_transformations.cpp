#include "meta/type_traits.hpp"

#include <type_traits>

template <class T>
constexpr bool cv_matches =
    meta::is_same_v<meta::remove_cv_t<T>, std::remove_cv_t<T>>;

static_assert(cv_matches<const volatile int>);
static_assert(cv_matches<const int *>);
static_assert(meta::is_same_v<meta::remove_pointer_t<int *const>,
                              std::remove_pointer_t<int *const>>);
static_assert(meta::is_same_v<meta::remove_pointer_t<const int *>,
                              std::remove_pointer_t<const int *>>);
static_assert(meta::is_same_v<meta::add_const_t<int>, std::add_const_t<int>>);
static_assert(meta::is_same_v<meta::add_pointer_t<int &>,
                              std::add_pointer_t<int &>>);
static_assert(meta::is_same_v<meta::add_lvalue_reference_t<void>,
                              std::add_lvalue_reference_t<void>>);
static_assert(meta::is_same_v<meta::add_rvalue_reference_t<int>,
                              std::add_rvalue_reference_t<int>>);

int main() {}
