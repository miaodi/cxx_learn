#include "meta/type_traits.hpp"

#include <type_traits>

using function = int(double);
using array = const int[4];

static_assert(meta::is_same_v<meta::remove_cvref_t<const volatile int &>,
                              std::remove_cvref_t<const volatile int &>>);
static_assert(meta::is_same_v<meta::decay_t<const int &>,
                              std::decay_t<const int &>>);
static_assert(meta::is_same_v<meta::decay_t<array>, std::decay_t<array>>);
static_assert(meta::is_same_v<meta::decay_t<function>,
                              std::decay_t<function>>);

int main() {}
