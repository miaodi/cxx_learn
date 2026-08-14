#include "meta/type_traits.hpp"

#include <type_traits>

template <class T>
constexpr bool pointer_matches = meta::is_pointer_v<T> == std::is_pointer_v<T>;

static_assert(pointer_matches<int>);
static_assert(pointer_matches<int *>);
static_assert(pointer_matches<const int *>);
static_assert(pointer_matches<int *const>);
static_assert(pointer_matches<int *volatile>);
static_assert(pointer_matches<int *&>);
static_assert(meta::is_void_v<const void> == std::is_void_v<const void>);
static_assert(meta::is_volatile_v<volatile int> ==
              std::is_volatile_v<volatile int>);
static_assert(meta::is_volatile_v<volatile int &> ==
              std::is_volatile_v<volatile int &>);

int main() {}
