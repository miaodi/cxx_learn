#include "meta/type_traits.hpp"

#include <string>
#include <vector>

struct no_members {};
struct addable {};

constexpr addable operator+(addable, addable) { return {}; }

static_assert(meta::has_value_type_v<std::vector<int>>);
static_assert(!meta::has_value_type_v<int>);
static_assert(meta::has_begin_v<std::string>);
static_assert(!meta::has_begin_v<no_members>);
static_assert(meta::is_addable_v<int, double>);
static_assert(meta::is_addable_v<addable, addable>);
static_assert(!meta::is_addable_v<no_members, no_members>);

int main() {}
