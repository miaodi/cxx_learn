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
  std::cout << std::left << std::setw(22) << label << " = " << std::right
            << std::setw(16)
            << std::setprecision(std::numeric_limits<float>::max_digits10) << value
            << "  raw=0x" << std::hex << std::uppercase << std::setw(8)
            << std::setfill('0') << float_bits(value) << std::dec << std::nouppercase
            << std::setfill(' ') << '\n';
}

void print_spacing(float value) {
  const float next = std::nextafter(value, std::numeric_limits<float>::infinity());
  const float delta = next - value;

  print_float("x", value);
  print_float("next larger float", next);
  print_float("delta", delta);
  std::cout << '\n';
}

int main() {
  std::cout << "Section 1: delta to the next larger float\n";
  std::cout << "For positive normal floats, this delta is nondecreasing as x grows.\n\n";
  print_spacing(1.0f);
  print_spacing(2.0f);
  print_spacing(1024.0f);
  print_spacing(16777216.0f);

  const float large = 16777216.0f;
  const float plus_one = large + 1.0f;
  const float plus_two = large + 2.0f;

  std::cout << "Section 2: adding below the current spacing\n";
  print_float("large", large);
  print_float("large + 1.0f", plus_one);
  print_float("large + 2.0f", plus_two);

  std::cout << std::boolalpha;
  std::cout << "large + 1.0f == large: " << (plus_one == large) << '\n';
  std::cout << "large + 2.0f == large: " << (plus_two == large) << '\n';
}
