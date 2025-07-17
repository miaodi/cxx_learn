#pragma once
#include <random>

void CrossProd(double const *a, double const *b, double *cross_prod);

void CrossProdAVX2(const double *a, const double *b, double *cross_prod);

double DotProd(double const *a, double const *b);

double DotProdAVX2(const double *a, const double *b);

double DotProdFMA(const double *a, const double *b);

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

struct alignas(32) AlignedDouble3 {
  double data[3];
  AlignedDouble3() : data{0.0, 0.0, 0.0} {}
  inline double &operator[](size_t index) { return data[index]; }
};
