#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <string_view>
#include <vector>

double naive_sum(std::span<const double> values) {
  double sum = 0.0;
  for (double value : values) {
    sum += value;
  }
  return sum;
}

double reverse_sum(std::span<const double> values) {
  double sum = 0.0;
  for (std::size_t i = values.size(); i > 0; --i) {
    sum += values[i - 1];
  }
  return sum;
}

double pairwise_sum(std::span<const double> values) {
  if (values.empty()) {
    return 0.0;
  }
  if (values.size() == 1) {
    return values.front();
  }

  const std::size_t middle = values.size() / 2;
  return pairwise_sum(values.first(middle)) + pairwise_sum(values.subspan(middle));
}

double kahan_sum(std::span<const double> values) {
  double sum = 0.0;
  double compensation = 0.0;

  for (double value : values) {
    // Carry the low-order part that was rounded out of the previous addition.
    const double corrected_value = value - compensation;
    const double next_sum = sum + corrected_value;
    compensation = (next_sum - sum) - corrected_value;
    sum = next_sum;
  }

  return sum;
}

long double reference_sum(std::span<const double> values) {
  long double sum = 0.0L;
  for (double value : values) {
    sum += static_cast<long double>(value);
  }
  return sum;
}

void print_double_row(std::string_view method, double result, long double exact) {
  const long double error = std::abs(static_cast<long double>(result) - exact);
  std::cout << std::left << std::setw(22) << method << std::right
            << std::setw(25) << std::setprecision(std::numeric_limits<double>::max_digits10)
            << result << "  error=" << std::scientific << std::setprecision(3)
            << error << std::defaultfloat << '\n';
}

void print_long_double_row(std::string_view method, long double result, long double exact) {
  const long double error = std::abs(result - exact);
  std::cout << std::left << std::setw(22) << method << std::right
            << std::setw(25)
            << std::setprecision(std::numeric_limits<long double>::max_digits10)
            << result << "  error=" << std::scientific << std::setprecision(3)
            << error << std::defaultfloat << '\n';
}

std::uint32_t float_bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

void print_float_value(std::string_view expression, float value) {
  std::cout << "  " << std::left << std::setw(22) << expression << " = "
            << std::right << std::setw(16)
            << std::setprecision(std::numeric_limits<float>::max_digits10) << value
            << "  raw=0x" << std::hex << std::uppercase << std::setw(8)
            << std::setfill('0') << float_bits(value) << std::dec << std::nouppercase
            << std::setfill(' ') << '\n';
}

void print_float_comparison(std::string_view title) {
  std::cout << title << '\n';
  std::cout << "Inputs:\n";
}

void print_float_result(std::string_view left_expression, float left,
                        std::string_view right_expression, float right) {
  print_float_value(left_expression, left);
  print_float_value(right_expression, right);
  std::cout << "  equal? " << std::boolalpha << (left == right) << "\n\n";
}

void reduction_section() {
  constexpr std::size_t small_count = 1u << 20;
  const double small = std::ldexp(1.0, -54);

  std::vector<double> values;
  values.reserve(small_count + 1);
  values.push_back(1.0);
  values.insert(values.end(), small_count, small);

  const long double exact = 1.0L + std::ldexp(static_cast<long double>(small_count), -54);
  const long double reference = reference_sum(values);

  std::cout << "Section 1: compensated reduction\n";
  std::cout << "Input: 1.0 followed by " << small_count << " copies of 2^-54\n";
  std::cout << "Each tiny term is exactly representable, but it is below half an ulp near 1.0.\n";
  std::cout << "Exact result: "
            << std::setprecision(std::numeric_limits<long double>::max_digits10)
            << exact << "\n\n";

  print_double_row("naive forward", naive_sum(values), exact);
  print_double_row("naive reverse", reverse_sum(values), exact);
  print_double_row("pairwise", pairwise_sum(values), exact);
  print_double_row("Kahan", kahan_sum(values), exact);
  print_long_double_row("long double ref", reference, exact);
}

void algebra_section() {
  std::cout << "\nSection 2: algebra rules that fail for float\n";
  std::cout << "The real-number identities are valid in mathematics, but each side rounds at different points.\n\n";

  {
    const float x = 100000000.0f;
    const float y = 1.0000001f;
    const float z = 1.0f;
    print_float_comparison("x * (y - z) != x * y - x * z");
    print_float_value("x", x);
    print_float_value("y", y);
    print_float_value("z", z);
    std::cout << "Results:\n";
    print_float_result("x * (y - z)", x * (y - z), "x * y - x * z",
                       x * y - x * z);
  }

  {
    const float x = 1.0e20f;
    const float y = -1.0e20f;
    const float z = 3.14f;
    print_float_comparison("(x + y) + z != x + (y + z)");
    print_float_value("x", x);
    print_float_value("y", y);
    print_float_value("z", z);
    std::cout << "Results:\n";
    print_float_result("(x + y) + z", (x + y) + z, "x + (y + z)",
                       x + (y + z));
  }

  {
    const float x = 5.0f;
    const float a = 3.0f;
    print_float_comparison("x / a != x * (1.0f / a)");
    print_float_value("x", x);
    print_float_value("a", a);
    std::cout << "Results:\n";
    print_float_result("x / a", x / a, "x * (1/a)", x * (1.0f / a));
  }
}

int main() {
  reduction_section();
  algebra_section();
}
