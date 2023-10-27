#ifndef multislice_lmastrategy_aux_h
#define multislice_lmastrategy_aux_h

#include "../Utility/Image.h"
#include "../Utility/Lattice.h"

#include "../CPU/ProbeApproximation.h"

#include <vector>
#include <array>
#include <set>

// Calculates the cost of adding the point p (given in probe lattice coordinates)
// to the computation domain, where certain Multislice solutions may be
// available for free as given by the zero_cost array.
int calculateCost(const std::array<int, 2>& p,
                  const Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                  const Lattice& input_wave_lattice,
                  const Lattice& probe_lattice,
                  const Image<bool>& zero_cost,
                  const Param& param) {
  int cost = 0;
  
  const int i = p[0] % CoefficientIndices.getX();
  const int j = p[1] % CoefficientIndices.getY();
  
  for (const auto& coeff_coord: CoefficientIndices(i, j)) {
    std::array<int, 2> iw_coord = getInputWaveCoord(p, coeff_coord, input_wave_lattice, probe_lattice, param);
    
    if (!zero_cost(iw_coord))
      ++cost;
  }
  
  return cost;
}

// Set the coordinates of all input waves required for the computation of
// the probe at position p to zero cost in the zero_cost array
void setZeroCost(const std::array<int, 2>& p,
                 const Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                 const Lattice& input_wave_lattice,
                 const Lattice& probe_lattice,
                 Image<bool>& zero_cost,
                 const Param& param) {
  const int i = p[0] % CoefficientIndices.getX();
  const int j = p[1] % CoefficientIndices.getY();
  
  for (const auto& coeff_coord: CoefficientIndices(i, j)) {
    std::array<int, 2> iw_coord = getInputWaveCoord(p, coeff_coord, input_wave_lattice, probe_lattice, param);
    
    zero_cost.set(iw_coord, true);
  }
}

// Calculates a vector of all input waves that need to be transmitted
// through the specimen for the computation of all probe positions in
// the probe_pos vector, shifted by (offset_x, offset_y). The probe
// positions are given with respect to the probe lattice and the input
// wave positions with respect to the input wave positions
std::vector<std::array<int, 2>> getRequiredInputWavePositions(const std::vector<std::array<int, 2>>& probe_pos,
                                                              const Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                                                              const Lattice& input_wave_lattice,
                                                              const Lattice& probe_lattice,
                                                              const int offset_x,
                                                              const int offset_y,
                                                              const Param& param) {
  // The input wave positions are first collected in a set to avoid duplicates
  auto cmp = [&](const std::array<int, 2>& a, const std::array<int, 2>& b) {
    return (a[0] + a[1] * input_wave_lattice.X < b[0] + b[1] * input_wave_lattice.X);
  };
  
  std::set<std::array<int, 2>, decltype(cmp)> iw_positions(cmp);
  
  for (std::array<int, 2> p: probe_pos) {
    p[0] += offset_x;
    p[1] += offset_y;
    
    const int i = p[0] % CoefficientIndices.getX();
    const int j = p[1] % CoefficientIndices.getY();
    
    for (std::array<int, 2> coeff_coord: CoefficientIndices(i, j)) {
      std::array<int, 2> iw_coord = getInputWaveCoord(p, coeff_coord, input_wave_lattice, probe_lattice, param);
      
      iw_positions.insert(iw_coord);
    }
  }
  
  // Convert to a vector
  return std::vector<std::array<int, 2>>(iw_positions.begin(), iw_positions.end());
}

#endif  // multislice_lmastrategy_aux_h
