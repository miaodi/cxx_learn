#pragma once
#include <cstdint>
#include <cmath>
#include <iostream>
#include <algorithm>
namespace MCRand {
std::uint64_t twiddle_origin(std::uint64_t u, std::uint64_t v);

std::uint64_t twiddle_new(std::uint64_t u, std::uint64_t v);

#define MT_N 624
#define MT_M 397

class tmcRand {
public:
  /**
   * Default seed value chosen from previous init_gen function
   */
  static constexpr std::uint32_t default_seed = 5489u;
  /**
   * @brief Initialize a pseudo-random number generator using
   */
  tmcRand(std::uint32_t seed = default_seed);

  // seeding for RNG
  void init_seed(std::uint32_t seed);

  // generates a random number on [0,1)-real-interval
  double drand() { // divided by 2^32
    return rand_int32() * (1.0 / 4294967296.0);
  }

private:
  void gen_state();
  static constexpr std::uint32_t num_states = MT_N, m = MT_M;
  std::uint32_t _p;
  std::uint32_t _states[num_states];

  // Generate a random 32-bit unsigned integer
  std::uint32_t rand_int32();
};

// generate 32 bit random int
inline std::uint32_t tmcRand::rand_int32() {
  // new state vector if needed
  if (_p == num_states)
    gen_state();

  auto y = _states[_p++];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9D2C5680U;
  y ^= (y << 15) & 0xEFC60000U;
  y ^= (y >> 18);

  return y;
}

#ifdef __AVX2__
class tmcRandAVX2 {
public:
  static constexpr std::uint32_t default_seed = 5489u;
  tmcRandAVX2(std::uint32_t seed = default_seed);

  void init_seed(std::uint32_t seed);

  // generates a random number on [0,1)-real-interval
  double drand() { // divided by 2^32
    return rand_int32() * (1.0 / 4294967296.0);
  }

private:
  void gen_state();
  static constexpr std::uint32_t num_states = MT_N, m = MT_M;
  std::uint32_t _p;
  alignas(32) std::uint32_t _states[num_states];
  inline std::uint32_t rand_int32() {
    // new state vector if needed
    if (_p == num_states)
      gen_state();

    auto y = _states[_p++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9D2C5680U;
    y ^= (y << 15) & 0xEFC60000U;
    y ^= (y >> 18);

    return y;
  }
};
#endif // __AVX2__


} // namespace MCRand
