// Auxiliary functions for calculating the approximations of the probes
// by linear combinations of other images (= input waves)
#ifndef multislice_cpu_probeapproximation_aux_h
#define multislice_cpu_probeapproximation_aux_h

#include "MSData.h"

#include "../Utility/Image.h"
#include "../Utility/Error.h"
#include "../Utility/Lattice.h"
#include "../Utility/AlignedArray.h"

#include <Eigen/Core>
#include <Eigen/SVD>

#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdlib>

// Disable internal parallelization of Eigen with OpenMP (because parallelization is
// handled on a higher level, in the function initProbeApproximation() in the file
// LatticeMultisliceAlgorithm_aux.h)
#define EIGEN_DONT_PARALLELIZE

// Finds a least squares approximation of target as a linear combination of the first N
// images in the vector imgs using Eigen
std::vector<complex> leastSquaresApprox(const std::vector<Image<complex>>& imgs,
                                        const int N,
                                        const Image<complex>& target) {
  const int X = target.getX();
  const int Y = target.getY();
  
  // Copy the input images to a complex valued Eigen matrix, where each column corresponds to one image
  Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> A(X*Y, N);
  
  for (int i=0; i<N; i++)
    for (int j=0; j<X*Y; j++)
      A.coeffRef(j, i) = imgs[i][j];
  
  // Copy target to an Eigen vector for the righthand side
  Eigen::Matrix<complex, Eigen::Dynamic, 1> b(X*Y);
  
  for (int j=0; j<X*Y; j++)
    b.coeffRef(j) = target[j];
  
  // Find a least squares solution
  Eigen::Matrix<complex, Eigen::Dynamic, 1> sol = A.colPivHouseholderQr().solve(b);
  
  // Convert to a std::vector of complex values
  std::vector<complex> coefficients(N);
  
  for (int i=0; i<N; i++)
    coefficients[i] = sol[i];
  
  return coefficients;
}

// Calculates the size, position and sub-pixel offset of a window with a side length of
// twice the probe radius centered at the given probe position and aligned to the grid
// given by the propagation window pixel size.
AlignedArray getProbeWindow(const MSData& data, const std::array<RealType, 2>& probe_pos) {
  const std::array<RealType, 2> grid_pixel_size = {data.propagation_window.pixel_size_x,
                                                   data.propagation_window.pixel_size_y};
    
  return AlignedArray(probe_pos, {0, 0}, data.probe_window_size, grid_pixel_size);
}

// Calculates the size, position and sub-pixel offset of the propagation window centered
// at the given input wave position and aligned to the grid given by the propagation
// window pixel size
AlignedArray getInputWaveWindow(const MSData& data,
                                const std::array<RealType, 2>& input_wave_pos,
                                const std::array<int, 2>& input_wave_index,
                                const std::array<RealType, 2>& probe_pos) {
  std::array<RealType, 2> iw_pos;
	std::array<int, 2> size;
	std::array<RealType, 2> grid_pixel_size;
	
	if (data.p.inputwave == MS_InputWave::FourierSpaceDirac) {
		iw_pos = probe_pos;
		size = {data.simulation_window.X, data.simulation_window.Y};
		grid_pixel_size = {data.simulation_window.pixel_size_x, data.simulation_window.pixel_size_y};
	} else {
		iw_pos = input_wave_pos;
		size = {data.propagation_window.X, data.propagation_window.Y};
		grid_pixel_size = {data.propagation_window.pixel_size_x, data.propagation_window.pixel_size_y};
	}
  
  return AlignedArray(iw_pos, input_wave_index, size, grid_pixel_size);
}

// Calculates the coordinates of the input wave that is used for the approximation of the
// probe at the position probe_coord in probe_lattice. The argument coeff_coord determines
// which of the (usually many) input waves that are used to approximate the probe at a
// given position is returned.
//
// The calling function should iterate over all coeff_coord in CoefficientIndices(i, j)
// when approximating the probe, where i = probe_coord[0] % CoefficientIndices.getX() and
// j = probe_coord[1] % CoefficientIndices.getY().
//
// See also the explanation of the CoefficientSets member of the class
// LatticeMultisliceAlgorithm in LatticeMultisliceAlgorithm.h
std::array<int, 2> getInputWaveCoord(const std::array<int, 2>& probe_coord,
                                     const std::array<int, 2>& coeff_coord,
                                     const Lattice& input_wave_lattice,
                                     const Lattice& probe_lattice,
                                     const Param& p) {
	// For the Fourier space Dirac input wave, coeff_coord needs to be
	// shifted by half of the input wave lattice dimensions, because it is
	// centered at the origin
	if (p.inputwave == MS_InputWave::FourierSpaceDirac)
	  return {coeff_coord[0] + input_wave_lattice.X / 2,
			      coeff_coord[1] + input_wave_lattice.Y / 2};
	
  const int A = input_wave_lattice.X;
  const int B = input_wave_lattice.Y;
  
  const int X = probe_lattice.X;
  const int Y = probe_lattice.Y;
  
  const int c1 = std::max(1, A / X);
  const int d1 = std::max(1, B / Y);
  
  const int c2 = std::max(1, X / A);
  const int d2 = std::max(1, Y / B);
  
  return {(coeff_coord[0] + (probe_coord[0] / c2) * c1) % input_wave_lattice.X,
          (coeff_coord[1] + (probe_coord[1] / d2) * d1) % input_wave_lattice.Y};
}

// Calculates the rectangular intersection of the probe and input wave windows,
// taking the periodicity of the simulation domain into account
void getOverlappingWindowArea(const MSData& data,
                              const AlignedArray& probe_window,
                              const AlignedArray& input_window,
                              std::array<int, 2>& probe_window_start,
                              std::array<int, 2>& input_window_start,
                              std::array<int, 2>& intersection_size) {
  if (data.p.inputwave == MS_InputWave::FourierSpaceDirac)
    Error("getOverlappingWindowArea() must not be called for the FourierSpaceDirac input wave type!", __FILE__, __LINE__);
  
  // Calculate the center of the probe and input wave windows in the
  // simulation domain in pixel
  const std::array<int, 2> pw_center = {static_cast<int>(normalize_periodic(probe_window.wave_position[0], data.simulation_window.lenX) / data.simulation_window.pixel_size_x),
                                        static_cast<int>(normalize_periodic(probe_window.wave_position[1], data.simulation_window.lenY) / data.simulation_window.pixel_size_y)};
  
  const std::array<int, 2> iw_center = {static_cast<int>(normalize_periodic(input_window.wave_position[0], data.simulation_window.lenX) / data.simulation_window.pixel_size_x),
                                        static_cast<int>(normalize_periodic(input_window.wave_position[1], data.simulation_window.lenY) / data.simulation_window.pixel_size_y)};
  
  // Shift the probe window by the width and/or height of the simulation
  // window as appropriate such that the probe center is as close as
  // possible to the input wave center (in absolute, non-periodic coordinates).
  //
  // Only the position of the top left corner of the probe window is
  // used in the following, so only these coordinates are changed and
  // not pw_center itself.
  std::array<int, 2> pw_pos = probe_window.aa_pos;
  const std::array<int, 2> iw_pos = input_window.aa_pos;
  
  const std::array<int, 2> dim = {data.simulation_window.X, data.simulation_window.Y};
  for (int i=0; i<2; i++) {
    const int diff = pw_center[i] - iw_center[i];
    
    const std::array<int, 3> dist = {std::abs(diff),
                                     std::abs(diff + dim[i]),
                                     std::abs(diff - dim[i])};
    
    if (dist[1] < dist[0] && dist[1] < dist[2])
      pw_pos[i] += dim[i];
    else if (dist[2] < dist[0] && dist[2] < dist[1])
      pw_pos[i] -= dim[i];
  }
  
  // Calculate the offset of the top left corners of the probe and input
  // wave windows; this may be negative or positive
  const std::array<int, 2> offset = {pw_pos[0] - iw_pos[0],
                                     pw_pos[1] - iw_pos[1]};
  
  for (int i=0; i<2; i++) {
    if (offset[i] < 0) {
      input_window_start[i] = 0;
      probe_window_start[i] = -offset[i];
    } else {
      input_window_start[i] = offset[i];
      probe_window_start[i] = 0;
    }
    
    intersection_size[i] = std::min(input_window.aa_size[i] - input_window_start[i],
                                    probe_window.aa_size[i] - probe_window_start[i]);
  }
}

// Calculates the norms of the difference of the probe and the linear combinations of the
// input waves with the coefficients given by coeff_vec, divided by the norm of the
// probe image.
//
// Note that input_waves may contain more elements than coeff_vec. In this case, only
// those elements for which there are coefficients for the linear combination are
// considered, i.e. the first coeff_vec.size() elements of input_waves.
void calculateApproximationErrors(const std::vector<Image<complex>>& input_waves,
                                  const std::vector<complex>& coeff_vec,
                                  const Image<complex>& probe,
                                  RealType *error_euc,
                                  RealType *error_sup) {
  Image<complex> res(probe.getX(), probe.getY(), {0, 0});
  
  const int N = static_cast<int>(coeff_vec.size());
  
  for (int j=0; j<N; j++) {
    Image<complex> tmp = input_waves[j];
    tmp *= coeff_vec[j];
    res += tmp;
  }
  
  Image<complex> diff(res);
  diff -= probe;
  
  *error_euc = norm(diff) / norm(probe);
  *error_sup = sup_norm(diff) / sup_norm(probe);
}

// Probe approximation step 1: calculate indices of all input waves
// within the probe radius around probe_coord that may be used in the
// least squares approximation
std::vector<std::array<int, 2>> getNearbyInputWaveIndices(const MSData& data,
                                                          const std::array<int, 2> probe_coord,
                                                          const Lattice& probe_lattice,
                                                          const Lattice& input_wave_lattice,
                                                          const bool generate_output,
                                                          const int max_num_input_waves) {
  // Calculate the probe position in Angstrom
  std::array<RealType, 2> probe_pos = probe_lattice.getPosition(probe_coord[0], probe_coord[1]);
  
  // Calculate the indices of all points of the input wave lattice that are within a
  // distance of data.p.probe_radius of the probe position
  std::vector<std::array<int, 2>> input_wave_indices;
  
  input_wave_indices = input_wave_lattice.getNearbyPointsPeriodic(probe_pos, data.p.probe_radius);
  
  if (generate_output) {
    #pragma omp critical (initProbeApproximation)
    {
      std::cerr << "\t(*) Number of input waves within the probe radius: " << input_wave_indices.size() << std::endl;
    }
  }
  
  if (max_num_input_waves != -1 &&
      static_cast<int>(input_wave_indices.size()) > max_num_input_waves) {
    
    input_wave_indices.resize(max_num_input_waves);
    
    if (generate_output) {
      #pragma omp critical (initProbeApproximation)
      {
        std::cerr << "\t(*) Reduced the number of input waves considered for the approximation to the" << std::endl
                  << "\t    " << max_num_input_waves << " input waves closest to the probe position" << std::endl
                  << "\t    (Parameter \"approximation_max_input_waves\")" << std::endl;
      }
    }
  }
  
  if (input_wave_indices.empty())
	Error("No input waves available for the approximation of the probe ("
	      + std::to_string(probe_coord[0]) + ", " + std::to_string(probe_coord[1]) + ")!");
  
  return input_wave_indices;
}

// Probe approximation step 2: calculate (potentially downsampled) images
// of the input waves for the least squares approximation
std::vector<Image<complex>> getNearbyInputWaveImages(const MSData& data,
                                                     const std::array<int, 2> probe_coord,
                                                     const Lattice& probe_lattice,
                                                     const Lattice& input_wave_lattice,
                                                     const std::vector<std::array<int, 2>>& input_wave_indices,
                                                     const bool generate_output) {
  // Calculate the probe position in Angstrom
  std::array<RealType, 2> probe_pos = probe_lattice.getPosition(probe_coord[0], probe_coord[1]);
  
  // Calculates the size and position of the probe window in pixel
  AlignedArray probe_window = getProbeWindow(data, probe_pos);
  
  // Calculate images of the input waves at all positions in the input_wave_indices vector
  std::vector<Image<complex>> input_waves;
  input_waves.reserve(input_wave_indices.size());
  
  for (const std::array<int, 2> index: input_wave_indices) {
    // Get the input wave coordinates in the input wave lattice
    // (Remark: here, iw_coord will always be equal to index)
    const std::array<int, 2> iw_coord = getInputWaveCoord(probe_coord, index, input_wave_lattice, probe_lattice, data.p);
    
    // Get the input wave position in Angstrom
    const std::array<RealType, 2> input_wave_pos = input_wave_lattice.getPosition(iw_coord[0], iw_coord[1]);
    
    // Get the input wave window coordinates
    AlignedArray input_window = getInputWaveWindow(data,
                                                   input_wave_pos,
                                                   index /* irrelevant */,
                                                   probe_pos /* irrelevant */);
    
    // Calculate the input wave on the full propagation window
    Image<complex> full_input_wave;
    
    full_input_wave = data.input.getInitialCondition(data.simulation_window.pixel_size_x,
                                                     data.simulation_window.pixel_size_y,
                                                     input_window.aa_size[0],
                                                     input_window.aa_size[1],
                                                     input_window.rel_wave_position[0],
                                                     input_window.rel_wave_position[1]);
    
    // Calculate the intersection of the probe and input wave windows
    std::array<int, 2> probe_window_start;
    std::array<int, 2> input_window_start;
    std::array<int, 2> intersection_size;
    
    getOverlappingWindowArea(data,
                             probe_window,
                             input_window,
                             probe_window_start,
                             input_window_start,
                             intersection_size);
    
    // Paste the input wave into the probe window
    Image<complex> input_wave(probe_window.aa_size[0], probe_window.aa_size[1], {0, 0});
    for (int y=0; y<intersection_size[1]; y++)
      for (int x=0; x<intersection_size[0]; x++)
        input_wave(probe_window_start[0] + x, probe_window_start[1] + y)
         = full_input_wave(input_window_start[0] + x, input_window_start[1] + y);
    
    // Add a subsampled version to the input waves vector
    input_waves.push_back(subsampleImage(input_wave, data.p.approximation_downsampling_factor));
  }
  
  // Save images of some of the input waves if requested
  if (generate_output && data.p.writeProbeApproximationOutput) {
    const int num_input_waves = static_cast<int>(input_waves.size());
    const int increment = std::max(num_input_waves/20, 1);
    for (int i=0; i<num_input_waves; i+=increment) {
      save(input_waves[i],
           std::to_string(i),
           data.p.outputDir + "/ProbeApproximation/InputWaves/RS_Re",
           data.p.outputDir + "/ProbeApproximation/InputWaves/RS_Im");
      
      Image<complex> input_wave_fs(input_waves[i]);
      FourierTransform(input_waves[i], &input_wave_fs);
      input_wave_fs.applyFourierShift();
      
      save(input_wave_fs,
           std::to_string(i),
           data.p.outputDir + "/ProbeApproximation/InputWaves/FS_Re",
           data.p.outputDir + "/ProbeApproximation/InputWaves/FS_Im");
    }
  }
  
  return input_waves;
}

// Probe approximation step 3: calculate a (potentially downsampled) image
// of the probe, which is to be approximated by the input waves
Image<complex> getTargetProbeImage(const MSData& data,
                                   const std::array<int, 2> probe_coord,
                                   const Lattice& probe_lattice,
                                   const bool generate_output) {
  // Calculate the probe position in Angstrom
  std::array<RealType, 2> probe_pos = probe_lattice.getPosition(probe_coord[0], probe_coord[1]);
  
  // Calculates the size and position of the probe window in pixel
  AlignedArray probe_window = getProbeWindow(data, probe_pos);
  
  // Calculate the probe image
  Image<complex> probe = data.input.probe.get2DImageNonSquare(data.simulation_window.pixel_size_x,
                                                              data.simulation_window.pixel_size_y,
                                                              probe_window.aa_size[0],
                                                              probe_window.aa_size[1],
                                                              probe_window.rel_wave_position[0],
                                                              probe_window.rel_wave_position[1],
                                                              {0, 0});
  
  probe = subsampleImage(probe, data.p.approximation_downsampling_factor);
  
  // Write probe to file if requested
  if (generate_output && data.p.writeProbeApproximationOutput) {
    save(probe, "Probe_RS", data.p.outputDir + "/ProbeApproximation/Probe");
    
    Image<complex> probe_fs(probe);
    FourierTransform(probe, &probe_fs);
    probe_fs.applyFourierShift();
    
    save(probe_fs, "Probe_FS", data.p.outputDir + "/ProbeApproximation/Probe");
  }
  
  return probe;
}

// Probe approximation step 4: perform multiple least squares approximations
// of the probe with an increasing number of input waves until the desired
// approximation error threshold is reached or all available input waves
// have been included in the approximation
void performLeastSquaresApproximations(const MSData& data,
                                       const Lattice& input_wave_lattice,
                                       const std::vector<std::array<int, 2>>& input_wave_indices,
                                       const std::vector<Image<complex>>& input_waves,
                                       const Image<complex> probe,
                                       Image<complex>& coeff,
                                       std::vector<std::array<int, 2>>& coeff_indices,
                                       RealType& approximation_error_euc,
                                       RealType& approximation_error_sup) {
  // Number of additional input waves to be added in the next iteration
  int num_waves_increment = 1;
  
  // num_waves_increment is doubles every num_waves_increment_steps iterations
  const int num_waves_increment_steps = 5;
  
  // Auxiliary variable to account for changes in num_waves_increment
  int offset = 0;
  
  // Minimum relative approximation errors in the supremum norm and the euclidean norm
  RealType min_approximation_error_sup = -1;
  RealType min_approximation_error_euc = -1;
  
  for (int N=0, step=0; N<static_cast<int>(input_waves.size()); N++) {
    if ((N+offset)%num_waves_increment != 0 && N+1 < static_cast<int>(input_waves.size()))
      continue;
    
    ++step;
    if (step == num_waves_increment_steps) {
      step = 0;
      num_waves_increment *= 2;
      offset = -N;
    }
    
    // Directly use all available input waves if the target approximation
    // error is zero.
    if (data.p.max_probe_approximation_error <= 0)
	  N = static_cast<int>(input_waves.size())-1;
    
    // Calculate the least squares approximation of probe by a linear combination of the
    // first N+1 elements in input waves
    std::vector<complex> coeff_vec = leastSquaresApprox(input_waves, N+1, probe);
    
    // Calculate approximation errors of the current least squares approximation
    calculateApproximationErrors(input_waves, coeff_vec, probe, &min_approximation_error_euc, &min_approximation_error_sup);
    
    // Stop if the approximation error in the supremum norm is below the requested
    // maximum error or if all input waves have been included in the approximation
    if (min_approximation_error_sup < data.p.max_probe_approximation_error ||
        N + 1 == static_cast<int>(input_waves.size())) {
      // Copy the result to the coefficient matrix and the corresponding indices to
      // coeff_indices
      coeff = Image<complex>(input_wave_lattice.X, input_wave_lattice.Y, {0, 0});
      coeff_indices.clear();
      
      for (int i=0; i<N+1; i++) {
        coeff(input_wave_indices[i][0], input_wave_indices[i][1]) = coeff_vec[i];
        coeff_indices.push_back(input_wave_indices[i]);
      }
      
      break;
    }
  }
  
  approximation_error_euc = min_approximation_error_euc;
  approximation_error_sup = min_approximation_error_sup;
}

#endif  // multislice_cpu_probeapproximation_aux_h
