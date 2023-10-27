// Definition of the complex data type and some auxiliary functions
#ifndef multislice_utility_complex_h
#define multislice_utility_complex_h

#include <complex>
#include <vector>

typedef std::complex<RealType> complex;

RealType abs_sqr(const complex& a) {
  return a.real() * a.real() + a.imag() * a.imag();
}

std::vector<complex> operator*(const RealType r, const std::vector<complex>& b) {
  std::vector<complex> res(b);
  for (int i=0; i<static_cast<int>(b.size()); i++)
    res[i] *= r;
  return res;
}


#endif  // multislice_utility_complex_h
