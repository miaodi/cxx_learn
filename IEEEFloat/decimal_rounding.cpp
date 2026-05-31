#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string_view>

static_assert(std::numeric_limits<float>::is_iec559);
static_assert(std::numeric_limits<double>::is_iec559);
static_assert(sizeof(float) == sizeof(std::uint32_t));
static_assert(sizeof(double) == sizeof(std::uint64_t));

std::uint32_t float_bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

std::uint64_t double_bits(double value) {
  return std::bit_cast<std::uint64_t>(value);
}

void print_float(std::string_view label, float value) {
  std::cout << std::left << std::setw(18) << label << " = " << std::right
            << std::setw(16)
            << std::setprecision(std::numeric_limits<float>::max_digits10) << value
            << "  raw=0x" << std::hex << std::uppercase << std::setw(8)
            << std::setfill('0') << float_bits(value) << std::dec << std::nouppercase
            << std::setfill(' ') << '\n';
}

void print_double(std::string_view label, double value) {
  std::cout << std::left << std::setw(18) << label << " = " << std::right
            << std::setw(24)
            << std::setprecision(std::numeric_limits<double>::max_digits10) << value
            << "  raw=0x" << std::hex << std::uppercase << std::setw(16)
            << std::setfill('0') << double_bits(value) << std::dec
            << std::nouppercase << std::setfill(' ') << '\n';
}

void print_decimal_error(std::string_view label, long double stored,
                         long double decimal_value) {
  std::cout << std::left << std::setw(18) << label << " stored - decimal = "
            << std::right << std::scientific << std::setprecision(6)
            << (stored - decimal_value) << std::defaultfloat << '\n';
}

void float_section() {
  const float tenth = 0.1f;
  const float two_tenths = 0.2f;
  const float three_tenths = 0.3f;
  const float sum = tenth + two_tenths;

  std::cout << "Section 1: decimal literals rounded to float\n";
  print_float("0.1f", tenth);
  print_float("0.2f", two_tenths);
  print_float("0.3f", three_tenths);
  print_float("0.1f + 0.2f", sum);

  std::cout << std::boolalpha;
  std::cout << "0.1f + 0.2f == 0.3f: " << (sum == three_tenths) << "\n\n";

  print_decimal_error("0.1f", tenth, 0.1L);
  print_decimal_error("0.2f", two_tenths, 0.2L);
  print_decimal_error("0.3f", three_tenths, 0.3L);
}

void double_section() {
  const double tenth = 0.1;
  const double two_tenths = 0.2;
  const double three_tenths = 0.3;
  const double sum = tenth + two_tenths;

  std::cout << "\nSection 2: the classic double comparison\n";
  print_double("0.1", tenth);
  print_double("0.2", two_tenths);
  print_double("0.3", three_tenths);
  print_double("0.1 + 0.2", sum);

  std::cout << std::boolalpha;
  std::cout << "0.1 + 0.2 == 0.3: " << (sum == three_tenths) << "\n\n";

  print_decimal_error("0.1", tenth, 0.1L);
  print_decimal_error("0.2", two_tenths, 0.2L);
  print_decimal_error("0.3", three_tenths, 0.3L);
}

int main() {
  float_section();
  double_section();
}
