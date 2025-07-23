#include "func.h"
#ifdef __AVX2__
#include <immintrin.h>
#endif

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

tmcRand::tmcRand(std::uint32_t seed) { init_seed(seed); }

// init by 32 bit seed
void tmcRand::init_seed(std::uint32_t s) {

  _states[0] = s; // for > 32 bit machines

  for (std::uint32_t i = 1; i < num_states; ++i) {
    _states[i] = 1812433253U * (_states[i - 1] ^ (_states[i - 1] >> 30)) + i;
    // see Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier
    // in the previous versions, MSBs of the seed affect only MSBs of the array
    // state 2002/01/09 modified by Makoto Matsumoto
  }

  // force gen_state() to be called for next random number
  _p = num_states;
}

std::uint32_t twiddle(std::uint32_t u, std::uint32_t v) {
  return (((u & UPPER_MASK) | (v & LOWER_MASK)) >> 1) ^ (-(v & 0x1) & MATRIX_A);
}

void tmcRand::gen_state() {
  for (std::uint32_t i = 0; i < num_states; ++i)
    _states[i] = _states[(i + m) % (num_states)] ^
                 twiddle(_states[i], _states[(i + 1) % (num_states)]);
  // reset position
  _p = 0;
}


#ifdef __AVX2__
tmcRandAVX2::tmcRandAVX2(std::uint32_t seed) { init_seed(seed); }

void tmcRandAVX2::init_seed(std::uint32_t seed) {
  _states[0] = seed;
  for (std::uint32_t i = 1; i < num_states; ++i) {
    _states[i] = 1812433253U * (_states[i - 1] ^ (_states[i - 1] >> 30)) + i;
  }
  _p = num_states; // force gen_state() to be called next
}

// AVX2 version: processes 8 uint32_t at a time
__m256i twiddle_avx2(const __m256i u, const __m256i v) {
  static const __m256i UPPER_MASK_VEC = _mm256_set1_epi32(0x80000000U);
  static const __m256i LOWER_MASK_VEC = _mm256_set1_epi32(0x7fffffffU);
  static const __m256i MATRIX_A_VEC = _mm256_set1_epi32(0x9908b0dfU);
  static const __m256i ONE_VEC = _mm256_set1_epi32(1);
  static const __m256i ZERO_VEC = _mm256_setzero_si256();

  __m256i u_upper = _mm256_and_si256(u, UPPER_MASK_VEC);
  __m256i v_lower = _mm256_and_si256(v, LOWER_MASK_VEC);
  __m256i merged = _mm256_or_si256(u_upper, v_lower);

  __m256i shifted = _mm256_srli_epi32(merged, 1);

  // v & 1
  __m256i v_lsb = _mm256_and_si256(v, ONE_VEC);
  // -(v & 1)
  __m256i v_lsb_neg = _mm256_sub_epi32(ZERO_VEC, v_lsb);
  // (-(v & 1)) & MATRIX_A
  __m256i matrix_mask = _mm256_and_si256(v_lsb_neg, MATRIX_A_VEC);

  return _mm256_xor_si256(shifted, matrix_mask);
}

void tmcRandAVX2::gen_state() {
  constexpr std::uint32_t vec_size = 8;
  std::uint32_t i = 0;
  alignas(32) std::uint32_t v_buf[vec_size];
  alignas(32) std::uint32_t mm_buf[vec_size];
  for (; i < num_states; i += vec_size) {
    __m256i u =
        _mm256_load_si256(reinterpret_cast<const __m256i *>(&_states[i]));
    for (std::uint32_t j = 0; j < vec_size; ++j) {
      v_buf[j] = _states[(i + 1 + j) % num_states];
      mm_buf[j] = _states[(i + m + j) % num_states];
    }
    __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i *>(v_buf));
    __m256i mm = _mm256_load_si256(reinterpret_cast<const __m256i *>(mm_buf));
    __m256i t = twiddle_avx2(u, v);
    __m256i res = _mm256_xor_si256(mm, t);
    _mm256_store_si256(reinterpret_cast<__m256i *>(&_states[i]), res);
  }
  _p = 0;
}
#endif // __AVX2__
} // namespace MCRand