#include <bit>
#include <bitset>
#include <climits>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

std::uint32_t encode_twos_complement(int value, int width) {
  if (width <= 0 || width > 31) {
    throw std::invalid_argument("width must be in [1, 31]");
  }

  // With N bits, two's complement can represent [-2^(N-1), 2^(N-1)-1].
  const int min_value = -(1 << (width - 1));
  const int max_value = (1 << (width - 1)) - 1;
  if (value < min_value || value > max_value) {
    throw std::out_of_range("value does not fit in the requested width");
  }

  // Interpret the N-bit pattern as an unsigned number modulo 2^N.
  // Non-negative values keep their ordinary binary representation.
  // Negative values wrap around: -5 in 8 bits is 256 - 5 = 251.
  const std::uint32_t modulus = 1u << width;
  return value >= 0 ? static_cast<std::uint32_t>(value)
                    : static_cast<std::uint32_t>(modulus + value);
}

std::string bits(std::uint32_t encoded, int width) {
  const std::string all_bits = std::bitset<32>(encoded).to_string();
  // Keep only the requested low bits, since the encoding width is part of the lesson.
  return all_bits.substr(all_bits.size() - static_cast<std::size_t>(width));
}

void show_encoding(int value, int width) {
  const std::uint32_t encoded = encode_twos_complement(value, width);

  std::cout << std::setw(4) << value << " encoded as " << std::setw(2) << width
            << " bits: 0b" << bits(encoded, width) << " = 0x" << std::hex
            << std::uppercase << encoded << std::dec << std::nouppercase << '\n';
}

template <typename Signed, typename Unsigned>
void show_native_object_bits(std::string_view type_name, Signed value) {
  static_assert(sizeof(Signed) == sizeof(Unsigned));

  // bit_cast reads the object representation without converting the value.
  // If the raw unsigned bits match the formula, this platform stores that
  // signed type using two's complement.
  const auto raw_bits = std::bit_cast<Unsigned>(value);
  const int width = static_cast<int>(sizeof(Signed) * CHAR_BIT);
  const std::uint32_t raw = static_cast<std::uint32_t>(raw_bits);
  const std::uint32_t expected =
      encode_twos_complement(static_cast<int>(value), width);

  std::cout << type_name << '(' << std::setw(4) << static_cast<int>(value)
            << ") object bits: 0b" << bits(raw, width) << " = 0x" << std::hex
            << std::uppercase << raw << std::dec << std::nouppercase
            << (raw == expected ? "  matches two's complement" : "  differs")
            << '\n';
}

int main() {
  std::cout << "Two's complement encodes a negative value x in N bits as 2^N + x.\n\n";

  constexpr int width = 8;
  // The negative examples show the top bit becoming the sign bit while the
  // remaining bits still participate in ordinary binary arithmetic.
  for (const int value : {5, 1, 0, -1, -2, -5, -128}) {
    show_encoding(value, width);
  }

  std::cout << "\nExample: -5 in 8 bits => 2^8 - 5 = 251 = 0b11111011.\n";

  std::cout << "\nInspecting actual std::int8_t object bits on this platform:\n";
  show_native_object_bits<std::int8_t, std::uint8_t>("int8_t ", -1);
  show_native_object_bits<std::int8_t, std::uint8_t>("int8_t ", -2);
  show_native_object_bits<std::int8_t, std::uint8_t>("int8_t ", -5);
}
