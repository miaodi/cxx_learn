#include "func.h"

namespace MCRand {

constexpr std::uint64_t UPPER_MASK = 0x80000000UL;
constexpr std::uint64_t LOWER_MASK = 0x7fffffffUL;
constexpr std::uint64_t MATRIX_A = 0x9908b0dfUL;
// Move tmcRand::twiddle to a free function
std::uint64_t twiddle_origin(std::uint64_t u, std::uint64_t v) {
  return (((u & UPPER_MASK) | (v & LOWER_MASK)) >> 1) ^
         (v & 0x1 ? MATRIX_A : 0);
}

// Move tmcRand::twiddle to a free function
std::uint64_t twiddle_new(std::uint64_t u, std::uint64_t v) {
  return (((u & UPPER_MASK) | (v & LOWER_MASK)) >> 1) ^ (-(v & 0x1) & MATRIX_A);
}
} // namespace MCRand