#pragma once
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#ifdef __AVX2__
#include <immintrin.h>
#endif // __AVX2__

namespace misc {

#define MCMinSize 1.0e-9

class MCVector {
public:
  MCVector() {
    ve[0] = 0;
    ve[1] = 0;
    ve[2] = 0;
  }
  MCVector(double a, double b, double c) {
    ve[0] = a;
    ve[1] = b;
    ve[2] = c;
  }
  MCVector(const MCVector &inV) {
    ve[0] = inV.ve[0];
    ve[1] = inV.ve[1];
    ve[2] = inV.ve[2];
  }
  MCVector &operator=(const MCVector &inV) {
    ve[0] = inV.ve[0];
    ve[1] = inV.ve[1];
    ve[2] = inV.ve[2];
    return *this;
  }
  MCVector &operator+=(const MCVector &a) {
    ve[0] += a[0];
    ve[1] += a[1];
    ve[2] += a[2];
    return *this;
  }
  MCVector &operator-=(const MCVector &a) {
    ve[0] -= a[0];
    ve[1] -= a[1];
    ve[2] -= a[2];
    return *this;
  }
  MCVector &operator*=(double a) {
    ve[0] *= a;
    ve[1] *= a;
    ve[2] *= a;
    return *this;
  }
  MCVector &operator/=(double a) {
    if (std::abs(a) < 1.0e-18) {
      if (a < 0)
        a -= 1.0e-18;
      else
        a += 1.0e-18;
    }
    ve[0] /= a;
    ve[1] /= a;
    ve[2] /= a;
    return *this;
  }

  inline bool operator<(const MCVector &a) const;

  double &operator[](int index) { return ve[index]; }
  const double &operator[](int index) const { return ve[index]; }
  void Set(double a, double b, double c) {
    ve[0] = a;
    ve[1] = b;
    ve[2] = c;
  }
  double length() const {
    double tmp = ve[0] * ve[0] + ve[1] * ve[1] + ve[2] * ve[2];
    if (tmp > 0.0)
      return std::sqrt(tmp);
    else
      return 0.0;
  }
  double getx() const { return ve[0]; }
  double gety() const { return ve[1]; }
  double getz() const { return ve[2]; }

  void setx(double val) { ve[0] = val; }
  void sety(double val) { ve[1] = val; }
  void setz(double val) { ve[2] = val; }

  // private:
  double ve[3];
};

std::ostream &operator<<(std::ostream &os, const MCVector &v);

/* ---------------------------------------------------------------*/

inline double operator*(const MCVector &a, const MCVector &b) {
  return (a[0]) * (b[0]) + (a[1]) * (b[1]) + (a[2]) * (b[2]);
}

inline MCVector operator^(const MCVector &a, const MCVector &b) {
  return MCVector((a[1]) * (b[2]) - (a[2]) * (b[1]),
                  (a[2]) * (b[0]) - (a[0]) * (b[2]),
                  (a[0]) * (b[1]) - (a[1]) * (b[0]));
}

inline double crossProductLen(const MCVector &a, const MCVector &b) {
  MCVector cross = a ^ b;
  return cross.length();
}

inline double eleProductSqLen(const MCVector &a, const MCVector &b) {
  const double x = a[0] * b[0];
  const double y = a[1] * b[1];
  const double z = a[2] * b[2];

  return x * x + y * y + z * z;
}

inline MCVector
operator*(double a, const MCVector &b) { // Implemented in terms of *=.  See
                                         // Meyers, More Effective C++, Item 22.
  return MCVector(b) *= a;
}

inline MCVector operator*(const MCVector &b,
                          double a) { // Implemented in terms of *=.  See
                                      // Meyers, More Effective C++, Item 22.
  return MCVector(b) *= a;
}

inline MCVector operator/(const MCVector &b,
                          double a) { // Implemented in terms of /=.  See
                                      // Meyers, More Effective C++, Item 22.
  return MCVector(b) /= a;
}

inline MCVector
operator+(const MCVector &a,
          const MCVector &b) { // Implemented in terms of +=.  See Meyers, More
                               // Effective C++, Item 22.
  return MCVector(a) += b;
}

inline MCVector
operator-(const MCVector &a,
          const MCVector &b) { // Implemented in terms of +=.  See Meyers, More
                               // Effective C++, Item 22.
  return MCVector(a) -= b;
}

inline MCVector operator-(const MCVector &a) {
  return MCVector(-(a[0]), -(a[1]), -(a[2]));
}

// Vector comparison
inline bool operator==(const MCVector &in1, const MCVector &in2) {
  return (std::abs(in1[0] - in2[0]) < MCMinSize) &&
         (std::abs(in1[1] - in2[1]) < MCMinSize) &&
         (std::abs(in1[2] - in2[2]) < MCMinSize);
}

// Vector comparison
inline bool operator!=(const MCVector &in1, const MCVector &in2) {
  return !(in1 == in2);
}

inline double distance(const MCVector &a, const MCVector &b) {
  double t = (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) +
             (a[2] - b[2]) * (a[2] - b[2]);
  return std::sqrt(t);
}

inline double distance2(const MCVector &a, const MCVector &b) {
  return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) +
         (a[2] - b[2]) * (a[2] - b[2]);
}

inline double dotProduct(const MCVector &a, const MCVector &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline MCVector crossProduct(const MCVector &a, const MCVector &b) {
  return MCVector(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                  a[0] * b[1] - a[1] * b[0]);
}

inline bool MCVector::operator<(const MCVector &a) const {
  // If two vectors are equal, one must not be less than the other.
  return ((ve[0] < a.ve[0]) || (!(a.ve[0] < ve[0]) && (ve[1] < a.ve[1])) ||
          (!(a.ve[0] < ve[0]) && !(a.ve[1] < ve[1]) && (ve[2] < a.ve[2]))) &&
         (*this != a);
}

#ifdef __AVX2__
class MCVectorAVX2 {
  union DATA {
    __m256d ve;
    double data[4];
  };

public:
  // constructors
  MCVectorAVX2() { _ve.ve = _mm256_setzero_pd(); }
  MCVectorAVX2(const double a, const double b, const double c) {
    _ve.ve = _mm256_set_pd(0.0, c, b, a);
  }
  MCVectorAVX2(const double a) { _ve.ve = _mm256_set1_pd(a); }
  MCVectorAVX2(const MCVectorAVX2 &other) : _ve(other._ve) {}
  MCVectorAVX2(const MCVector &other) {
    _ve.ve = _mm256_set_pd(0.0, other[2], other[1], other[0]);
  }
  MCVectorAVX2(const double *a) { _ve.ve = _mm256_loadu_pd(a); }
  MCVectorAVX2(const __m256d v) : _ve{v} {}

  // assignment operator
  MCVectorAVX2 &operator=(MCVectorAVX2 other) {
    std::swap(other._ve, _ve);
    return *this;
  }

  MCVectorAVX2 &operator=(const MCVector &other) {
    _ve.ve = _mm256_set_pd(0.0, other[2], other[1], other[0]);
    return *this;
  }

  // addition and subtraction
  MCVectorAVX2 &operator+=(const MCVectorAVX2 &other) {
    _ve.ve = _mm256_add_pd(_ve.ve, other._ve.ve);
    return *this;
  }
  MCVectorAVX2 &operator-=(const MCVectorAVX2 &other) {
    _ve.ve = _mm256_sub_pd(_ve.ve, other._ve.ve);
    return *this;
  }

  // multiplication and division
  MCVectorAVX2 &operator*=(const double a) {
    _ve.ve = _mm256_mul_pd(_ve.ve, _mm256_set1_pd(a));
    return *this;
  }

  MCVectorAVX2 &operator/=(double a) {
    if (std::abs(a) < 1.0e-18) {
      if (a < 0)
        a -= 1.0e-18;
      else
        a += 1.0e-18;
    }
    _ve.ve = _mm256_div_pd(_ve.ve, _mm256_set1_pd(a));
    return *this;
  }

  //   bool operator<(const MCVectorAVX2 &other) const {
  //     return _ve.ve < other._ve.ve;
  //   }

  MCVectorAVX2 operator-() const {
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    __m256d tmp = _mm256_xor_pd(_ve.ve, sign_mask);
    return MCVectorAVX2(tmp);
  }

  double operator*(const MCVectorAVX2 &other) const {
    __m256d tmp = _mm256_mul_pd(_ve.ve, other._ve.ve);
    _mm256_store_pd(_tmp, tmp);
    return _tmp[0] + _tmp[1] + _tmp[2];
  }

  MCVectorAVX2 operator^(const MCVectorAVX2 &other) const {
    // Permute: [a1, a2, a0, a3] and [b1, b2, b0, b3]
    __m256d va_yzx = _mm256_permute4x64_pd(_ve.ve, _MM_SHUFFLE(3, 0, 2, 1));
    __m256d vb_yzx =
        _mm256_permute4x64_pd(other._ve.ve, _MM_SHUFFLE(3, 0, 2, 1));

    // Compute cross product using only 3 permutations
    __m256d mul1 = _mm256_mul_pd(_ve.ve, vb_yzx);
    __m256d mul2 = _mm256_mul_pd(other._ve.ve, va_yzx);
    __m256d result = _mm256_sub_pd(mul1, mul2);

    // Final permutation to get the correct order: [res1, res2, res0, res3]
    return MCVectorAVX2(_mm256_permute4x64_pd(result, _MM_SHUFFLE(3, 0, 2, 1)));
  }

  double crossProductLen(const MCVectorAVX2 &other) const {

    // Permute: [a1, a2, a0, a3] and [b1, b2, b0, b3]
    __m256d va_yzx = _mm256_permute4x64_pd(_ve.ve, _MM_SHUFFLE(3, 0, 2, 1));
    __m256d vb_yzx =
        _mm256_permute4x64_pd(other._ve.ve, _MM_SHUFFLE(3, 0, 2, 1));

    // Compute cross product using only 3 permutations
    __m256d mul1 = _mm256_mul_pd(_ve.ve, vb_yzx);
    __m256d mul2 = _mm256_mul_pd(other._ve.ve, va_yzx);
    __m256d result = _mm256_sub_pd(mul1, mul2);
    __m256d tmp = _mm256_mul_pd(result, result);

    _mm256_store_pd(_tmp, tmp);
    return std::sqrt(_tmp[0] + _tmp[1] + _tmp[2]);
  }

  MCVectorAVX2 neg() const {
    __m256d tmp = _mm256_sub_pd(_mm256_set1_pd(0.0), _ve.ve);
    return MCVectorAVX2(tmp);
  }

  const double &operator[](int index) const { return _ve.data[index]; }
  double &operator[](int index) { return _ve.data[index]; }

  double length() const {
    __m256d tmp = _mm256_mul_pd(_ve.ve, _ve.ve);
    _mm256_store_pd(_tmp, tmp);
    return std::sqrt(_tmp[0] + _tmp[1] + _tmp[2]);
  }

  double eleProductSqLen(const MCVectorAVX2 &other) const {
    __m256d tmp = _mm256_mul_pd(_ve.ve, other._ve.ve);

    __m256d tmp2 = _mm256_mul_pd(tmp, tmp);

    _mm256_store_pd(_tmp, tmp2);
    return _tmp[0] + _tmp[1] + _tmp[2];
  }

  void set(double const *a) { _ve.ve = _mm256_loadu_pd(a); }

  void Set(const double a, const double b, const double c) {
    _ve.ve = _mm256_set_pd(0.0, c, b, a);
  }

  // getters and setters, less efficient than operator[]
  double getx() const { return _ve.data[0]; }
  double gety() const { return _ve.data[1]; }
  double getz() const { return _ve.data[2]; }

  void setx(double val) { _ve.data[0] = val; }
  void sety(double val) { _ve.data[1] = val; }
  void setz(double val) { _ve.data[2] = val; }

  // private:
  DATA _ve;

  alignas(32) mutable double _tmp[4];
};

inline MCVectorAVX2 operator*(MCVectorAVX2 a, const double b) { return a *= b; }
inline MCVectorAVX2 operator*(const double a, MCVectorAVX2 b) { return b * a; }

inline MCVectorAVX2 operator/(MCVectorAVX2 a, const double b) { return a /= b; }

inline MCVectorAVX2 operator+(MCVectorAVX2 a, const MCVectorAVX2 &b) {
  return a += b;
}

inline MCVectorAVX2 operator-(MCVectorAVX2 a, const MCVectorAVX2 &b) {
  return a -= b;
}

inline double dotProduct(const MCVectorAVX2 &a, const MCVectorAVX2 &b) {
  return a * b;
}

#endif // __AVX2__

template <typename Iterator>
void FillRandom(Iterator begin, Iterator end, double min = -1.0,
                double max = 1.0) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine{rnd_device()};
  std::uniform_real_distribution<double> dist{min, max};
  auto gen = [&]() { return dist(mersenne_engine); };
  for (auto it = begin; it != end; ++it) {
    (*it)[0] = gen();
    (*it)[1] = gen();
    (*it)[2] = gen();
  }
}
} // namespace misc