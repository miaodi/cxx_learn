#include <iostream>
#include <string>
#include <utility>

struct Expr {
  std::string text;
};

Expr expr(std::string text) {
  return Expr{std::move(text)};
}

Expr operator-(Expr lhs, Expr rhs) {
  return Expr{"(" + lhs.text + " - " + rhs.text + ")"};
}

template <typename... Ts>
auto right_fold_without_init(Ts... args) {
  return (args - ...);
}

template <typename... Ts>
auto left_fold_without_init(Ts... args) {
  return (... - args);
}

template <typename... Ts>
auto right_fold_with_init(Ts... args) {
  return (args - ... - 100);
}

template <typename... Ts>
auto left_fold_with_init(Ts... args) {
  return (100 - ... - args);
}

template <typename... Ts>
Expr right_fold_with_init_shape(Ts... args) {
  return (args - ... - expr("init"));
}

template <typename... Ts>
Expr left_fold_with_init_shape(Ts... args) {
  return (expr("init") - ... - args);
}

template <typename... Ts>
auto sum_or_zero(Ts... args) {
  return (0 + ... + args);
}

int main() {
  std::cout << "Fold expressions reduce a parameter pack into one expression.\n";
  std::cout << "Using operator- makes fold direction visible.\n\n";

  std::cout << "Integer values: 1, 2, 3, 4; binary-fold init: 100\n";
  std::cout << "  (args - ...)       right fold without init = "
            << right_fold_without_init(1, 2, 3, 4) << "\n";
  std::cout << "  (... - args)       left fold without init  = "
            << left_fold_without_init(1, 2, 3, 4) << "\n";
  std::cout << "  (args - ... - 100) right fold with init    = "
            << right_fold_with_init(1, 2, 3, 4) << "\n";
  std::cout << "  (100 - ... - args) left fold with init     = "
            << left_fold_with_init(1, 2, 3, 4) << "\n\n";

  std::cout << "Expression-shape tracer for args: a, b, c, d\n";
  std::cout << "  (args - ...)          -> "
            << right_fold_without_init(expr("a"), expr("b"), expr("c"),
                                       expr("d"))
                   .text
            << "\n";
  std::cout << "  (... - args)          -> "
            << left_fold_without_init(expr("a"), expr("b"), expr("c"),
                                      expr("d"))
                   .text
            << "\n";
  std::cout << "  (args - ... - init)   -> "
            << right_fold_with_init_shape(expr("a"), expr("b"), expr("c"),
                                          expr("d"))
                   .text
            << "\n";
  std::cout << "  (init - ... - args)   -> "
            << left_fold_with_init_shape(expr("a"), expr("b"), expr("c"),
                                         expr("d"))
                   .text
            << "\n\n";

  std::cout << "Empty packs need an identity value unless the operator has a "
               "built-in empty-pack rule.\n";
  std::cout << "  sum_or_zero()         -> " << sum_or_zero() << "\n";
  std::cout << "  sum_or_zero(1,2,3,4)  -> " << sum_or_zero(1, 2, 3, 4)
            << "\n\n";

  std::cout << "Knowledge point 1: there are four fold forms: right fold "
               "without init, left fold without init, right fold with init, "
               "and left fold with init.\n";
  std::cout << "Knowledge point 2: the direction matters for non-associative "
               "operators such as operator-.\n";
  std::cout << "Knowledge point 3: folds with init provide an initial value, "
               "which can also make empty packs well-defined.\n";
}
