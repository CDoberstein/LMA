#ifndef multislice_utility_atom_h
#define multislice_utility_atom_h

#include <tuple>

// Container for a single atom
struct Atom {
  int Z;                            // atomic number
  RealType base_x, base_y, base_z;  // base atom coordinates in Angstrom before applying random thermal motion
  RealType x, y, z;                 // current atom coordinates in Angstrom after applying random thermal motion
  RealType thermal_sigma;           // standard deviation of random thermal motion in Angstrom
  
  Atom(int Z, RealType x, RealType y, RealType z, RealType thermal_sigma)
    : Z(Z), base_x(x), base_y(y), base_z(z), x(x), y(y), z(z), thermal_sigma(thermal_sigma) {}
  
  bool operator<(const Atom& rhs) const {
    // Lexicographical order
    return std::tie(Z, base_x, base_y, base_z, thermal_sigma) < std::tie(rhs.Z, rhs.base_x, rhs.base_y, rhs.base_z, rhs.thermal_sigma);
  }
};

#endif  // multislice_utility_atom_h
