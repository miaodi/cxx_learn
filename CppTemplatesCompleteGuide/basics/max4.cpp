#include <iostream>

namespace non_template_after_three_arg_template {

template <typename T>
T max(T a, T b) {
  std::cout << "  max<T>(T,T)\n";
  return b < a ? a : b;
}

template <typename T>
T max(T a, T b, T c) {
  std::cout << "  max<T>(T,T,T), defined before max(int,int)\n";
  return max(max(a, b), c);
}

// This overload exists before run() instantiates max<int>(int,int,int), but it
// is too late for ordinary unqualified lookup inside that template definition.
int max(int a, int b) {
  std::cout << "  max(int,int)\n";
  return b < a ? a : b;
}

int run() {
  return max(47, 11, 33);
}

}  // namespace non_template_after_three_arg_template

namespace non_template_before_three_arg_template {

template <typename T>
T max(T a, T b) {
  std::cout << "  max<T>(T,T)\n";
  return b < a ? a : b;
}

int max(int a, int b) {
  std::cout << "  max(int,int)\n";
  return b < a ? a : b;
}

template <typename T>
T max(T a, T b, T c) {
  std::cout << "  max<T>(T,T,T), defined after max(int,int)\n";
  return max(max(a, b), c);
}

int run() {
  return max(47, 11, 33);
}

}  // namespace non_template_before_three_arg_template

int main() {
  std::cout << "Case 1: non-template max(int,int) is declared too late\n";
  const int late_result = non_template_after_three_arg_template::run();
  std::cout << "  result = " << late_result << "\n\n";

  std::cout << "Case 2: non-template max(int,int) is visible first\n";
  const int early_result = non_template_before_three_arg_template::run();
  std::cout << "  result = " << early_result << "\n\n";

  std::cout << "Knowledge point 1: when a non-template overload and a "
               "function template are both visible and equally good, overload "
               "resolution prefers the non-template overload.\n";
  std::cout << "Knowledge point 2: ordinary unqualified lookup inside a "
               "template body only sees declarations that are already visible "
               "at the template definition.\n";
}
