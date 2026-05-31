#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

static_assert(std::numeric_limits<float>::is_iec559);
static_assert(sizeof(float) == sizeof(std::uint32_t));

std::uint32_t float_bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

void print_float(std::string_view label, float value) {
  std::cout << std::left << std::setw(28) << label << " = " << std::right
            << std::setw(18)
            << std::setprecision(std::numeric_limits<float>::max_digits10) << value
            << "  raw=0x" << std::hex << std::uppercase << std::setw(8)
            << std::setfill('0') << float_bits(value) << std::dec << std::nouppercase
            << std::setfill(' ') << '\n';
}

void print_long_double(std::string_view label, long double value) {
  std::cout << std::left << std::setw(28) << label << " = " << std::right
            << std::setw(26)
            << std::setprecision(std::numeric_limits<long double>::max_digits10)
            << value << '\n';
}

void nearby_subtraction_section() {
  const float a = 1.0000001f;
  const float b = 1.0f;
  const float difference = a - b;

  std::cout << "Section 1: subtracting nearby rounded values\n";
  print_float("a", a);
  print_float("b", b);
  print_float("a - b", difference);
  std::cout << "Decimal intent was near 1e-7, but the stored float values decide the result.\n\n";
}

void stable_reformulation_section() {
  const float x = 100000000.0f;
  const float direct = std::sqrt(x + 1.0f) - std::sqrt(x);
  const float stable = 1.0f / (std::sqrt(x + 1.0f) + std::sqrt(x));

  const long double reference_x = 100000000.0L;
  const long double reference =
      1.0L / (std::sqrt(reference_x + 1.0L) + std::sqrt(reference_x));

  std::cout << "Section 2: stable algebraic reformulation\n";
  print_float("x", x);
  print_float("sqrt(x + 1) - sqrt(x)", direct);
  print_float("1 / (sqrt(x + 1)+sqrt(x))", stable);
  print_long_double("long double reference", reference);

  const long double direct_error = std::abs(static_cast<long double>(direct) - reference);
  const long double stable_error = std::abs(static_cast<long double>(stable) - reference);
  print_long_double("direct abs error", direct_error);
  print_long_double("stable abs error", stable_error);
}

int main() {
  nearby_subtraction_section();
  stable_reformulation_section();
}
