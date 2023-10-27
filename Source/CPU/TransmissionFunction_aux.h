// Auxiliary functions for the initialization of the TransmissionFunction struct
#ifndef multislice_cpu_transmissionfunction_aux_h
#define multislice_cpu_transmissionfunction_aux_h

#include "../Utility/CircularImage.h"
#include "../Utility/potential_coeff.h"

#include <vector>
#include <cmath>

// Calculates the interaction constant sigma
RealType getSigma(const RealType lambda, const RealType AcceleratingVoltage) {
  const RealType c = 299792458;           // Speed of light in meters per second
  const RealType e = 1.60217662e-19;      // Proton charge in Coulomb
  const RealType me = 9.10938356e-31;     // Electron rest mass in kg
  const RealType pi = 3.14159265359;
  
  const RealType eU = e * AcceleratingVoltage;
  const RealType mec2 = me * c * c;
  const RealType sigma = 2*pi / (lambda * AcceleratingVoltage) * ((mec2 + eU) / (2*mec2 + eU));
  
  return sigma;
}

// Calculate the projected potential of a given atomic species as a circular image
CircularImage<RealType> getProjectedAtomicPotential(const int atomic_number,
                                                    const RealType potential_radius,
                                                    const RealType pixel_size,
                                                    const RealType min_core_dist) {
  const RealType pi = 3.14159265359;
  const RealType a0 = 0.529;
  const RealType e = 14.4;
  const RealType coeff = 2 * pi * pi * a0 * e;
  
  const int X = static_cast<int>(potential_radius / pixel_size) + 1;
  std::vector<RealType> line(X);
  
  for (int x=0; x<X; x++) {
    // Calculate the distance to the atom center according to x and make sure, that the
    // distance is never zero because of the singularity of the projected potential
    const RealType r = std::max(x * pixel_size, min_core_dist);
    
    // Calculate the projected potential according to the formula given by Kirkland
    //
    // Note: atomic numbers start from 1, but the array index starts from 0
    const auto& pc = potential_coeff[atomic_number-1];
    
    line[x] = 2 * coeff * (pc[0] * std::cyl_bessel_k(0, 2*pi*r*sqrt(pc[1])) +
                           pc[2] * std::cyl_bessel_k(0, 2*pi*r*sqrt(pc[3])) +
                           pc[4] * std::cyl_bessel_k(0, 2*pi*r*sqrt(pc[5])))
                + coeff * ( pc[6] /  pc[7] * exp(-pi*pi*r*r/ pc[7]) +
                            pc[8] /  pc[9] * exp(-pi*pi*r*r/ pc[9]) +
                           pc[10] / pc[11] * exp(-pi*pi*r*r/pc[11]));
  }
  
  // Subtract the smallest value of the projected potential, line.back(), to ensure a
  // smooth transition to the zero potential of the vacuum surrounding the atoms
  for (int x=0; x<X; x++)
    line[x] -= line[X-1];
  
  return CircularImage<RealType>(line, pixel_size);
}

// Computes the atomic transmission functions exp(i*sigma*projected_potential(x))
CircularImage<complex> getAtomicTransmissionFunction(const CircularImage<RealType>& atom_potential, const Param& p) {
  const RealType sigma = getSigma(p.lambda, p.AcceleratingVoltage);
  
  std::vector<RealType> atom_potential_line = atom_potential.getData();
  
  std::vector<complex> line(atom_potential_line.size());
  for (unsigned int i=0; i<line.size(); i++)
    line[i] = complex(cos(sigma * atom_potential_line[i]),
                      sin(sigma * atom_potential_line[i]));
  
  return CircularImage<complex>(line, atom_potential.getPixelSize());
}

// Assigns all atoms in the specimen to their containing slice
std::vector<std::vector<int>> assignAtomsToSlices(const Specimen& specimen,
                                                  const RealType slice_thickness) {
  const int num_slices = static_cast<int>(ceil(specimen.lenZ / slice_thickness));
  
  std::vector<std::vector<int>> mapping(num_slices);
  for (int i=0; i<static_cast<int>(specimen.atoms.size()); i++) {
    const int slice_ind = std::clamp(static_cast<int>(specimen.atoms[i].z / specimen.lenZ * num_slices), 0, num_slices-1);
    mapping[slice_ind].push_back(i);
  }
  
  if (mapping.back().empty())
    mapping.pop_back();
  
  return mapping;
}

// Calculate the projected potential for one slice
Image<RealType> getProjectedPotentialSlice(const std::vector<Image<RealType>>& atom_potentials,
                                           const Specimen& specimen,
                                           const std::vector<int>& atom_indices,
                                           const std::vector<int>& atomic_number_map,
                                           const RealType simulation_pixel_size_x,
                                           const RealType simulation_pixel_size_y,
                                           const int X,
                                           const int Y) {
  Image<RealType> projected_potential(X, Y, static_cast<RealType>(0));
  
  // Iterate over all atoms in the slice and add their precomputed projected potential to
  // the overall projected potential for this slice using nearest neighbor interpolation
  for (const int i: atom_indices) {
    const Atom cur_atom = specimen.atoms[i];
    const Image<RealType> atom_pot = atom_potentials[atomic_number_map[cur_atom.Z]];
    
    const RealType atom_lenX = simulation_pixel_size_x * atom_pot.getX();
    const RealType atom_lenY = simulation_pixel_size_y * atom_pot.getY();
    
    const int x_start = static_cast<int>((cur_atom.x - atom_lenX/2) / simulation_pixel_size_x);
    const int y_start = static_cast<int>((cur_atom.y - atom_lenY/2) / simulation_pixel_size_y);
    
    const int x_end = x_start + atom_pot.getX();
    const int y_end = y_start + atom_pot.getY();
    
    const int x_left_offset = std::max(0, -x_start);
    const int y_left_offset = std::max(0, -y_start);
    
    const int x_right_offset = std::max(0, x_end - X);
    const int y_right_offset = std::max(0, y_end - Y);
    
    for (int y=y_left_offset; y<atom_pot.getY()-y_right_offset; y++)
      for (int x=x_left_offset; x<atom_pot.getX()-x_right_offset; x++) {
        projected_potential(x_start + x, y_start + y) += atom_pot(x, y);
      }
  }
  
  // Symmetrically bandwidth-limit the projected potential
  const RealType maxFreqX = 1 / (2 * simulation_pixel_size_x);
  const RealType maxFreqY = 1 / (2 * simulation_pixel_size_y);
  const RealType tmp = std::min(maxFreqX, maxFreqY);
  
  applyBandwidthLimit(&projected_potential, tmp / maxFreqX, tmp / maxFreqY);
  
  return projected_potential;
}

// Calculates the transmission function exp(i*sigma*pot(x))
Image<complex> getSliceTransmissionFunction(const Image<RealType>& pot,
                                            const Param& p,
                                            const RealType lenX,
                                            const RealType lenY) {
  const RealType sigma = getSigma(p.lambda, p.AcceleratingVoltage);
  
  const int X = pot.getX();
  const int Y = pot.getY();
  
  Image<complex> res(X, Y, {0, 0});
  
  for (int y=0; y<Y; y++)
    for (int x=0; x<X; x++)
      res(x, y) = complex(cos(sigma * pot(x, y)),
                          sin(sigma * pot(x, y)));
  
  // Symmetrically bandwidth-limit the transmission function
  // (see Kirkland, Advanced Computing in Electron Microscopy, 2010, p. 151)
  const RealType maxFreqX = X / (2 * lenX);
  const RealType maxFreqY = Y / (2 * lenY);
  const RealType tmp = std::min(maxFreqX, maxFreqY);
  
  applyBandwidthLimit(&res, 2/3. * tmp / maxFreqX, 2/3. * tmp / maxFreqY);
  
  return res;
}

#endif  // multislice_cpu_transmissionfunction_aux_h
