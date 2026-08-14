#include "expr_ad/expr_ad.hpp"

struct Label {};

// Expected failure: Label is neither an Expression nor an arithmetic value.
auto unsupported = expr_ad::x + Label{};

int main() {}
