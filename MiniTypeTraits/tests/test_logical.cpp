#include "meta/type_traits.hpp"

#include <type_traits>

struct incomplete;

template <class T>
struct requires_complete : meta::bool_constant<(sizeof(T) > 0)> {};

static_assert(meta::conjunction_v<> == std::conjunction_v<>);
static_assert(meta::conjunction_v<meta::true_type, meta::false_type> ==
              std::conjunction_v<std::true_type, std::false_type>);
static_assert(meta::disjunction_v<> == std::disjunction_v<>);
static_assert(meta::disjunction_v<meta::false_type, meta::true_type> ==
              std::disjunction_v<std::false_type, std::true_type>);
static_assert(meta::negation_v<meta::true_type> ==
              std::negation_v<std::true_type>);

// These compile only if conjunction and disjunction short-circuit.
static_assert(!meta::conjunction<meta::false_type,
                                 requires_complete<incomplete>>::value);
static_assert(meta::disjunction<meta::true_type,
                                requires_complete<incomplete>>::value);

int main() {}
