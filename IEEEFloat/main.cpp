#include <bit>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

static_assert(std::numeric_limits<float>::is_iec559);
static_assert(sizeof(float) == sizeof(std::uint32_t));

std::string bits(std::uint32_t value, int first, int count) {
  const std::string all_bits = std::bitset<32>(value).to_string();
  return all_bits.substr(static_cast<std::size_t>(first),
                         static_cast<std::size_t>(count));
}

void show_float32(std::string_view name, float value) {
  // bit_cast observes the float object representation without changing the value.
  const std::uint32_t raw_bits = std::bit_cast<std::uint32_t>(value);

  const std::string sign = bits(raw_bits, 0, 1);
  const std::string exponent = bits(raw_bits, 1, 8);
  const std::string fraction = bits(raw_bits, 9, 23);

  std::cout << std::left << std::setw(10) << name << ": " << sign << " | "
            << exponent << " | " << fraction << "  raw=0x" << std::hex
            << std::uppercase << std::right << std::setw(8) << std::setfill('0')
            << raw_bits << std::dec << std::nouppercase << std::setfill(' ')
            << std::left << '\n';
}

void explain_finite_value(std::string_view name, float value) {
  constexpr int bias = 127;
  constexpr int fraction_bits = 23;

  const std::uint32_t raw_bits = std::bit_cast<std::uint32_t>(value);
  const std::uint32_t encoded_exponent = (raw_bits >> fraction_bits) & 0xffu;
  const std::uint32_t fraction = raw_bits & 0x7fffffu;

  show_float32(name, value);

  if (encoded_exponent == 0) {
    // Subnormal numbers do not have the implicit leading 1 before the fraction.
    const int exponent = 1 - bias;
    const long double significand =
        static_cast<long double>(fraction) / (1u << fraction_bits);
    std::cout << "  subnormal: e=0, E=1-bias=" << exponent
              << ", M=0.fraction=" << std::setprecision(10) << significand
              << ", value=M*2^E=" << static_cast<float>(std::ldexp(significand, exponent))
              << "\n\n";
    return;
  }

  // Normal numbers have an implicit leading 1 before the stored fraction bits.
  const int exponent = static_cast<int>(encoded_exponent) - bias;
  const long double significand =
      1.0L + static_cast<long double>(fraction) / (1u << fraction_bits);
  std::cout << "  normal:    e=" << encoded_exponent << ", E=e-bias=" << exponent
            << ", M=1.fraction=" << std::setprecision(10) << significand
            << ", value=M*2^E=" << static_cast<float>(std::ldexp(significand, exponent))
            << "\n\n";
}

void signed_zero_section() {
  const float positive_zero = 0.0f;
  const float negative_zero = -0.0f;

  std::cout << "Section 1: signed zero\n";
  std::cout << "IEEE-754 float32 layout: sign | exponent | fraction\n\n";
  show_float32("+0.0f", positive_zero);
  show_float32("-0.0f", negative_zero);

  std::cout << std::boolalpha;
  std::cout << "\n+0.0f == -0.0f: " << (positive_zero == negative_zero) << '\n';
  std::cout << "signbit(+0.0f): " << std::signbit(positive_zero) << '\n';
  std::cout << "signbit(-0.0f): " << std::signbit(negative_zero) << '\n';

  std::cout << "\n1.0f / +0.0f: " << (1.0f / positive_zero) << '\n';
  std::cout << "1.0f / -0.0f: " << (1.0f / negative_zero) << '\n';
}

void infinity_and_nan_section() {
  const float positive_inf = std::numeric_limits<float>::infinity();
  const float negative_inf = -std::numeric_limits<float>::infinity();
  const float quiet_nan = std::numeric_limits<float>::quiet_NaN();

  std::cout << "\nSection 2: infinity and NaN\n";
  std::cout << "Exponent 11111111 has special meanings.\n\n";
  show_float32("+inf", positive_inf);
  show_float32("-inf", negative_inf);
  show_float32("NaN", quiet_nan);

  std::cout << std::boolalpha;
  std::cout << "\nisinf(+inf): " << std::isinf(positive_inf) << '\n';
  std::cout << "isnan(NaN): " << std::isnan(quiet_nan) << '\n';
  std::cout << "NaN == NaN: " << (quiet_nan == quiet_nan) << '\n';
}

void normal_and_subnormal_section() {
  const float normal = 1.5f;
  const float smallest_normal = std::numeric_limits<float>::min();
  const float smallest_subnormal = std::numeric_limits<float>::denorm_min();
  const float normal_fraction_one = std::bit_cast<float>(0x00800001u);
  const float subnormal_fraction_one = std::bit_cast<float>(0x00000001u);

  std::cout << "\nSection 3: normal and subnormal finite values\n";
  std::cout << "Normal uses M=1.fraction and E=e-bias.\n";
  std::cout << "Subnormal uses M=0.fraction and E=1-bias.\n\n";

  explain_finite_value("1.5f", normal);
  explain_finite_value("min norm", smallest_normal);
  explain_finite_value("min sub", smallest_subnormal);

  std::cout << "Same stored fraction field, different exponent field:\n";
  explain_finite_value("e=1 frac=1", normal_fraction_one);
  explain_finite_value("e=0 frac=1", subnormal_fraction_one);

  const std::uint32_t normal_bits = std::bit_cast<std::uint32_t>(normal_fraction_one);
  const std::uint32_t subnormal_bits = std::bit_cast<std::uint32_t>(subnormal_fraction_one);
  const std::uint32_t normal_fraction_bits = normal_bits & 0x7fffffu;
  const std::uint32_t subnormal_fraction_bits = subnormal_bits & 0x7fffffu;

  const long double normal_fraction =
      static_cast<long double>(normal_fraction_bits) / (1u << 23);
  const long double subnormal_fraction =
      static_cast<long double>(subnormal_fraction_bits) / (1u << 23);
  const long double normal_m = 1.0L + normal_fraction;
  const long double subnormal_m = subnormal_fraction;

  std::cout << "Ignoring the common 2^-126 factor: normal M - subnormal M = "
            << std::setprecision(10) << normal_m << " - " << subnormal_m
            << " = " << (normal_m - subnormal_m) << '\n';
}

int main() {
  signed_zero_section();
  infinity_and_nan_section();
  normal_and_subnormal_section();
}
