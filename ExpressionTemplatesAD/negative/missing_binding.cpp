#include "expr_ad/expr_ad.hpp"

// Expected failure: Variable<1> needs at least two supplied values.
auto missing = expr_ad::evaluate(expr_ad::y, 4.0);

int main() {}
