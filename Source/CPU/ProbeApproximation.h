// Functions for the approximation of a probe wave function by a linear combination of
// other functions
#ifndef multislice_cpu_probeapproximation_h
#define multislice_cpu_probeapproximation_h

#include "ProbeApproximation_aux.h"

#include "MSData.h"
#include "PropagationWindow.h"

#include "../Utility/Image.h"
#include "../Utility/AlignedArray.h"
#include "../Utility/Lattice.h"
#include "../Utility/Error.h"

#include <vector>
#include <complex>
#include <cmath>
#include <utility>
#include <array>
#include <algorithm>

// Computes the coefficients for the approximation of a probe by a linear combination of
// the input waves given by data.input.initial_condition. The resulting coefficients are
// stored in coeff and their indices in the input wave lattice are stored in coeff_indices.
//
// Also returns the relative approximation errors in the euclidean norm and the supremum norm.
void ProbeApproximation(const MSData& data,
                        const std::array<int, 2> probe_coord,
                        const Lattice& probe_lattice,
                        const Lattice& input_wave_lattice,
                        Image<complex>& coeff,
                        std::vector<std::array<int, 2>>& coeff_indices,
                        const bool generate_output,
                        const int max_num_input_waves,
                        RealType& approximation_error_euc,
                        RealType& approximation_error_sup) {
  // The probe coordinates must not exceed the quotient of probe lattice
  // and input wave lattice dimension because otherwise the input wave
  // lattice indices calculated in the functions below would need to be
  // shifted inversely such that calls to getInputWaveCoord() return the
  // indices exactly as returned by getNearbyPointsPeriodic(). If
  // probe_coord is within the interal [0, probe_lattice.X / input_wave_lattice.X),
  // then getInputWaveCoord(probe_coord, index, input_wave_lattice, probe_lattice, data.p)
  // simply returns index unchanged.
  if (probe_coord[0] < 0 || probe_coord[0] >= probe_lattice.X / input_wave_lattice.X ||
      probe_coord[1] < 0 || probe_coord[1] >= probe_lattice.Y / input_wave_lattice.Y) {
    Error("Invalid probe coordinates (" + std::to_string(probe_coord[0]) + ", " + std::to_string(probe_coord[1]) + ") in ProbeApproximation()!", __FILE__, __LINE__);
  }
  
  // Calculate indices of input waves near the probe that may be used
  // in the approximation according to parameters in data.p
  auto input_wave_indices = getNearbyInputWaveIndices(data,
                                                      probe_coord,
                                                      probe_lattice,
                                                      input_wave_lattice,
                                                      generate_output,
                                                      max_num_input_waves);
  
  // Calculate images of all input waves corresponding to the indices
  // in input_wave_indices
  auto input_waves = getNearbyInputWaveImages(data,
                                              probe_coord,
                                              probe_lattice,
                                              input_wave_lattice,
                                              input_wave_indices,
                                              generate_output);
  
  // Calculate an image of the probe at probe_coord
  auto probe = getTargetProbeImage(data,
                                   probe_coord,
                                   probe_lattice,
                                   generate_output);
  
  // Find a least squares approximation of the probe image by linear
  // combinations of the input wave images
  performLeastSquaresApproximations(data,
                                    input_wave_lattice,
                                    input_wave_indices,
                                    input_waves,
                                    probe,
                                    coeff,
                                    coeff_indices,
                                    approximation_error_euc,
                                    approximation_error_sup);
}

// Estimates the memory consumption of one call of the function
// ProbeApproximation() above in MB
RealType estimateProbeApproximationMB(const MSData& data,
                                      const Lattice& input_wave_lattice,
                                      const int max_num_input_waves) {
  // Number of input waves used for the least squares approximation
  int num_input_waves = static_cast<int>(input_wave_lattice.getNearbyPointsPeriodic({0, 0}, data.p.probe_radius).size());
  
  if (max_num_input_waves != -1)
    num_input_waves = std::min(max_num_input_waves, num_input_waves);
  
  // Size of the input wave (and probe) images in the least squares approximation
  AlignedArray probe_window = getProbeWindow(data, {0, 0});
  
  // Size of the input_waves vector in bytes: there is a total number of
  // num_input_waves elements with probe_window.aa_size complex pixels,
  // which are all downsampled according to downsampling_factor
  RealType s = num_input_waves;
  s *= probe_window.aa_size[0] * probe_window.aa_size[1];
  s *= sizeof(complex);
  s /= data.p.approximation_downsampling_factor * data.p.approximation_downsampling_factor;
  
  // The input_waves vector is duplicated in leastSquaresApprox() and we assume that
  // "A.colPivHouseholderQr().solve(b)" requires an additional s bytes
  return 3 * s / (1024 * 1024);
}

#endif  // multislice_cpu_probeapproximation_h
