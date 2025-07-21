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


#define MCMinSize 1.0e-9

class MCVector
{
 public:
   MCVector() {
      ve[0]=0; ve[1]=0; ve[2]=0;
   }
   MCVector(double a, double b, double c) {
      ve[0]=a; ve[1]=b; ve[2]=c;
   }
   MCVector(const MCVector& inV) {
      ve[0]=inV.ve[0]; ve[1]=inV.ve[1]; ve[2]=inV.ve[2];
   }
   MCVector& operator=(const MCVector& inV) {
      ve[0]=inV.ve[0]; ve[1]=inV.ve[1]; ve[2]=inV.ve[2];
      return *this;
   }
   MCVector& operator+=(const MCVector& a) {
      ve[0]+=a[0];
      ve[1]+=a[1];
      ve[2]+=a[2];
      return *this;
   }
   MCVector& operator-=(const MCVector& a) {
      ve[0]-=a[0];
      ve[1]-=a[1];
      ve[2]-=a[2];
      return *this;
   }
   MCVector& operator*=(double a) {
      ve[0]*=a;
      ve[1]*=a;
      ve[2]*=a;
      return *this;
   }
   MCVector& operator/=(double a) {
      if ( std::abs(a) < 1.0e-18 ) {
         if (a<0) a -= 1.0e-18;
         else     a += 1.0e-18;
      }
      ve[0]/=a;
      ve[1]/=a;
      ve[2]/=a;
      return *this;
   }
   bool operator<(const MCVector& a) const;
   double& operator[](int index) {return ve[index];}
   const double& operator[](int index) const {return ve[index];}
   void Set(double a, double b, double c) {
      ve[0]=a; ve[1]=b; ve[2]=c;
   }
   double length() const {
      double tmp = ve[0]*ve[0]+ve[1]*ve[1]+ve[2]*ve[2];
      if (tmp > 0.0) return std::sqrt(tmp);
      else return 0.0;
   }
   double getx() const {return ve[0];}
   double gety() const {return ve[1];}
   double getz() const {return ve[2];}

   void setx(double val){ve[0] = val;}
   void sety(double val){ve[1] = val;}
   void setz(double val){ve[2] = val;}
   
 private:
   double ve[3];
};

std::ostream& operator<<(std::ostream& os, const MCVector& v);

/* ---------------------------------------------------------------*/

inline
double operator*(const MCVector& a, const MCVector& b) 
{
   return (a[0])*(b[0])+(a[1])*(b[1])+(a[2])*(b[2]); 
}

inline
MCVector operator^(const MCVector& a ,const MCVector& b) 
{
   return MCVector((a[1])*(b[2])-(a[2])*(b[1]),
		 (a[2])*(b[0])-(a[0])*(b[2]), 
		 (a[0])*(b[1])-(a[1])*(b[0]));
} 

inline
MCVector operator*(double a, const MCVector& b) 
{  // Implemented in terms of *=.  See Meyers, More Effective C++, Item 22.
   return MCVector(b) *= a;
}

inline
MCVector operator*(const MCVector& b, double a) 
{  // Implemented in terms of *=.  See Meyers, More Effective C++, Item 22.
   return MCVector(b) *= a;
}

inline
MCVector operator/(const MCVector& b,double a) 
{  // Implemented in terms of /=.  See Meyers, More Effective C++, Item 22.
   return MCVector(b) /= a;
}

inline
MCVector operator+(const MCVector& a ,const MCVector& b) 
{  // Implemented in terms of +=.  See Meyers, More Effective C++, Item 22.
   return MCVector(a) += b;
}

inline
MCVector operator-(const MCVector& a ,const MCVector& b) 
{  // Implemented in terms of +=.  See Meyers, More Effective C++, Item 22.
   return MCVector(a) -= b;
}

inline
MCVector operator-(const MCVector& a) 
{
   return MCVector(-(a[0]),-(a[1]),-(a[2]));
} 

// Vector comparison
inline
bool operator==(const MCVector& in1, const MCVector& in2)
{
   return
      (std::abs(in1[0] - in2[0]) < MCMinSize) &&
      (std::abs(in1[1] - in2[1]) < MCMinSize) &&
      (std::abs(in1[2] - in2[2]) < MCMinSize);
}

// Vector comparison
inline
bool operator!=(const MCVector& in1, const MCVector& in2)
{
   return !(in1 == in2);
}

inline
double distance(const MCVector& a, const MCVector& b)
{
   double t = (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]);
   return std::sqrt(t);
}

inline
double distance2(const MCVector& a, const MCVector& b)
{
   return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]);
}
} // namespace MCRand
