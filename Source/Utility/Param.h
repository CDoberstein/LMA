// A container class for all program parameters
#ifndef multislice_utility_param_h
#define multislice_utility_param_h

#include "Param_aux.h"

#include "Specimen.h"

#include <omp.h>

#include <cmath>

#include <string>
#include <vector>
#include <array>
#include <utility>
#include <fstream>
#include <ctime>
#include <filesystem>

// The multislice algorithm may be performed on the CPU or on the GPU(s)
// (The GPU implementation is not included on Github)
enum class MS_Device { CPU, GPU };

/* The multislice algorithm may be performed in one of two ways:
 *
 *   (Fourierspace) alternatingly in real space and Fourier space coordinates, where the
 *                  transmission step is computed as usual as a pointwise multiplication
 *                  of the slice potential with the electron wave function in real space
 *                  and the propagation step is performed as a pointwise multiplication of
 *                  the electron wave with the propagation kernel in Fourier space.
 *                  This option is only available if MS_Device is set to CPU.
 *
 *   (Realspace) entirely in real space coordinates, performing the propagation step
 *               directly as a convolution in real space coordinates and avoiding the
 *               Fourier transform.
 */
enum class MS_Domain { FourierSpace, RealSpace };

/* Two variants of the propagator function are available for the multislice computations:
 * 
 *   (Full) use the standard propagator function given by P(k) = exp(-i*Pi*lambda*|k|^2*t)
 *          in Fourier space, where k is the 2D frequency, lambda is the electron
 *          wavelength and t is the slice thickness. Note that the corresponding
 *          propagator function in real space, i.e. the inverse Fourier transform of P, is
 *          given by p(x) = 1/(i*lambda*t) * exp(i*Pi/(lambda*t) * |x|^2), where x is the
 *          2D spatial position.
 * 
 *   (LowRankApproximation) use a low rank approximation to the real space propagator
 *                          function p, given by a set of functions f_1, ..., f_k such
 *                          that p(x,y) is approximately equal to
 *                            f_1(x) * f_1(y) + f_2(x) * f_2(y) + ... + f_k(x) * f_k(y).
 *                          This accelerates the convolutions in real space since the 2D
 *                          convolution with p is replaced by a sequence of 1D
 *                          convolutions with f_1, ..., f_k.
 */
enum class MS_Propagator { Full, LowRankApproximation };

/* There are several different kinds of input waves that may be transmitted through the
 * specimen. In a post-processing step, appropriate linear combinations of the solutions
 * to the Schrödinger equation obtained for these input waves are calculated to get the
 * solutions for the probe wave function initial condition. (This works because of the
 * linearity of the Schrödinger equation.)
 * 
 * The following types of input waves are implemented (see also CPU/InputWave.h):
 * 
 *   (Probe) This is the standard STEM simulation algorithm. For every probe position x, one
 *           probe wave function centered at x is transmitted through the specimen to
 *           obtain the corresponding pixel value of the STEM image directly.
 * 
 *   (ProbeSubset) The dimension of the probe vector space is at most equal to the number
 *                 of pixels M within the objective aperture in Fourier space. If N is the
 *                 number of pixels of the resulting STEM image (i.e. the total number of
 *                 probe positions), then M may be much smaller than N. If ProbeSubset is
 *                 chosen as the input wave type, then K probes will be transmitted
 *                 through the specimen, corresponding to probe positions on a standard
 *                 rectangular lattice L with the smallest number of points K such that
 *                 M <= K <= N or as given by the approximation_max_input_waves and
 *                 input_wave_lattice_r parameters below (see initPositionLattices() in
 *                 LatticeMultisliceAlgorithm_aux.h for the construction of the input wave
 *                 lattice)
 * 
 *   (TrigonometricPolynomial) Uses the trigonometric polynomial defined by
 * 
 *                                 g_n(x) = sum(exp(i*k*x)*cos(k*Pi/(2*n+2)), k=-n...n)
 * 
 *                             with degree n calculated from the maximum frequency of
 *                             the probe wave functions (see getTrigPolyFunction() in
 *                             CPU/InputWave_aux.h)
 * 
 *   (TrigonometricPolynomialV2) Just like TrigonometricPolynomial, but cropped in
 *                               Fourier space to frequencies within the objective
 *                               aperture, i.e. limited to frequencies of the probe
 *                               wave function
 * 
 *   (Square) Indicator function of a square
 * 
 *   (Disk) Indicator function of a disk
 * 
 *   (Pixel) Indicator function of a single pixel in the simulation window
 *   
 *   (Gaussian) A two-dimensional Gaussian function
 *   
 *   (FourierSpaceDirac) Exponential functions corresponding to single frequencies
 *                       in Fourier space (similar to the input functions used in PRISM)
 */
enum class MS_InputWave { Probe,
                          ProbeSubset,
                          TrigonometricPolynomial,
                          TrigonometricPolynomialV2,
                          Square,
                          Disk,
                          Pixel,
                          Gaussian,
                          FourierSpaceDirac };

// Container class for all microscope, simulation and program settings
struct Param {
  /** Microscope and probe settings **/
  // Accelerating voltage in Volts
  RealType AcceleratingVoltage;
  
  // Electron wavelength in Angstrom
  //
  // Note: not read from file but automatically calculated from the accelerating voltage
  RealType lambda;
  
  // Focus in Angstrom
  RealType Z;
  
  // Spherical aberration in Angstrom
  RealType Cs;
  
  // Maximum semiangle allowed by the objective aperture in radians
  RealType alpha_max;
  
  
  /** Specimen settings **/
  // Path to the .xyz specimen file
  std::string specimen_path;
  
  // Tiling parameters in x, y and z directions
  int tile_x, tile_y, tile_z;
  
  
  /** Simulation mode **/
  // Device: CPU or GPU
  MS_Device device;
  
  // Simulation domain: Fourierspace or Realspace
  MS_Domain domain;
  
  // Propagator type: Full or LowRankApproximation
  MS_Propagator propagator;
  
  // Type of the initial conditions to the Schrödinger equation
  MS_InputWave inputwave;
  
  // Number of iterations for the frozen phonon approximation
  // The default value is 1, which means that only one configuration without
  // any atom vibrations is computed.
  int frozen_phonon_iterations;
  
  
  /** Multislice settings **/
  // Requested pixel size of one pixel in the Multislice algorithm (in Angstrom)
  //
  // Note: the actual pixel size may slightly differ from the value given here to ensure
  //       that the size of the simulation window, which is equal to the specimen size,
  //       is an even integer multiple of the simulation pixel size (this only applies to
  //       Fourier space algorithms, i.e. if domain == MS_Domain::FourierSpace)
  RealType req_simulation_pixel_size_x, req_simulation_pixel_size_y;
  
  // Radius of the probe in Angstrom
  // This is used for precomputing probe images and determines the maximum
  // size of the propagation window in the real space algorithms as well as the
  // maximum size of the Multislice results in both the Fourier space and the
  // real space algorithms. Therefore, this value should be sufficiently large
  // to allow the probe and input waves to "spread out" while passing through
  // the specimen slices in the Multislice algorithm.
  RealType probe_radius;
  
  // Propagation window size in Angstrom
  // This value is clamped to [2*<input wave radius>, 4*<probe radius>]
  RealType propagation_window_size;
  
  // The number of exemplary Multislice simulations that are run before
  // the Lattice Multislice Algorithm for the sole purpose to determine
  // if the propagation window size is sufficient. Results from these
  // tests are saved to the PropagationWindowTest directory in the
  // output directory and must be manually inspected.
  // Default value: 5
  int propagation_window_test;
  
  // Thickness of the slices in Angstrom
  RealType slice_thickness;
  
  // Maxium distance from the atom center for precomputing the potential in Angstrom
  RealType potential_bound;
  
  // Radius stepsize of the virtual detectors for 3D STEM output given in mrad
  RealType detector_stepsize;
  
  // Supersampling factor for high-resolution circular images in the MSData struct
  int hires_supersampling_factor;
  
  // Maximum rank of the low-rank approximation to the propagator function
  int propagator_rank;
  
  // Lower bound for the radius of the real space propagation function in Angstrom
  RealType min_propagator_radius;
  
  // Size of the buffer zone used for the propagation window in the Fourier space
  // CPU implementation of the Multislice algorithm, relative to the propagation
  // window size. Example: if fs_buffer_zone = 0.5, then the buffer zone around the
  // propagation window has a width resp. height of half of the propagation window
  // size, effectively doubling the size of the propagation window in both
  // directions.
  // 
  // Negative values indicate to use the standard algorithm where the Fourier space
  // Multislice algorithm is performed on the entire simulation window.
  // 
  // [The wrap around error seems to have a negligible effect here, so that it
  //  may always be advisable to use fs_buffer_zone = 0.0 instead of -1]
  // 
  // Default value: -1
  RealType fs_buffer_zone;
  
  
  /** Simulation window settings **/
  // Requested probe step size in Angstrom
  // 
  // Note: the actual probe step size will be chosen close to these values such that the
  //       specimen width is an integer multiple of the probe step size. If input_wave
  //       is not equal to MS_InputWave::Probe and the number of points in the input wave
  //       lattice is smaller than the number of points in the probe lattice, then the
  //       probe_step size may be further adjusted to ensure that the two lattices are
  //       compatible (see initPositionLattices() in LatticeMultisliceAlgorithm_aux.h)
  RealType req_probe_step_x, req_probe_step_y;
  
  // Fractional coordinates for the subsection of the specimen that is contained in the
  // final output image
  std::array<RealType, 2> simulation_bounds_x, simulation_bounds_y;
  
  
  /** Input wave settings **/
  // Maximum error for the approximation of the probe wave functions as linear
  // combinations of the input wave functions, measured in the supremum norm and scaled
  // to the maximum value of the probe wave function image
  RealType max_probe_approximation_error;
  
  // Maximum number of input waves considered for the approximation of a probe wave
  // function. The default value is -1, which means that all input waves within a distance
  // of probe_radius of the probe center may be used.
  int approximation_max_input_waves;
  
  // Factor, by which the images in the least squares probe approximation are downsampled
  // in both directions to shorten the computation time. Default value: 2
  int approximation_downsampling_factor;
  
  // Square root of the factor by which the number of input waves is smaller than the
  // number of probe positions, as counted on the entire specimen domain. The default value is
  // -1, which means that the maximum value of r is automatically computed, such that the
  // number of input waves is still greater than or equal to the maximum probe vector
  // space dimension (this is used if inputwave is not MS_InputWave::Probe or MS_InputWave::Pixel)
  //
  // If inputwave == MS_InputWave::FourierSpaceDirac, then this corresponds to
  // PRISM's f parameter
  int input_wave_lattice_r;
  
  // Degree of the trigonometric polynomials if inputwave is TrigonometricPolynomial
  // or TrigonometricPolynomialV2. The default value is -1, which means that
  // a reasonable value for the degree is automatically computed based on the
  // frequencies in the probe wave function (which are limited by the objective
  // aperture)
  int trig_poly_degree;
  
  // Side length of the square input wave in Angstrom (if inputwave == MS_InputWave::Square)
  // Default value: 2.0
  RealType input_wave_square_len;
  
  // Radius of the disk input wave in Angstrom (if inputwave == MS_InputWave::Disk)
  // Default value: 1.0
  RealType input_wave_disk_radius;
  
  // Minimum propagation window size for the pixel input wave in Angstrom (if inputwave == MS_InputWave::Pixel)
  // Default value: 1.0
  RealType input_wave_pixel_propagation_window_size;
  
  // Standard deviation of the gaussian input wave in Angstrom (if inputwave == MS_InputWave::Gaussian)
  // Default value: 0.0 (i.e. a reasonable value is automatically calculated)
  RealType input_wave_gaussian_sigma;
  
  // If true, the idealized probe frequency values exp(-i*<aberration function>)
  // are used as coefficients in the linear combination approximating the STEM
  // probe. Otherwise the actual frequencies of the STEM probes as computed in
  // this program are used.
  // Default value: true
  bool input_wave_fsdirac_use_ideal_frequencies;
  
  // Number of pixels in Fourier space beyond the aperture radius to be taken
  // into consideration for the probe approximation (Fourier space frequencies
  // of the STEM probes beyond the aperture radius will be nonzero due to the
  // discretization)
  // This value must not be different from zero if input_wave_fsdirac_use_ideal_frequencies
  // is true.
  // Default value: 0
  int input_wave_fsdirac_px_beyond_aperture;
  
  // Only used in the approximation test when simulating STEM images with
  // varying numbers of input waves used to approximate the probe wave function.
  // Effectively equivalent to changing the approximation_max_input_waves
  // parameter accordingly.
  int test_max_num_input_waves_start;
  int test_max_num_input_waves_decrement;
  
  
  /** Output settings **/
  // Output directory
  std::string outputDir;
  
  // Determines if data from the initialization of the multislice algorithm is saved to
  // outputDir
  bool writeInitData;
  
  // Determines if the full slice potential and transmission functions are written to file
  // (ignored if writeInitData is false)
  //
  // Note: depending on the specimen and simulation pixel size, the slices may require
  //       many GB of disk space
  bool writeSlices;
  
  // Determines if some output is generated for the probe approximation step in the
  // initialization (ignored if inputwave == MS_InputWave::Probe)
  bool writeProbeApproximationOutput;
  
  // Determines if output from the search for a good LMA strategy is generated
  bool writeLMAStrategySearchOutput;
  
  // Determines if partial results are written to file after completing a
  // computation domain in the LMA strategy
  bool writePartialSTEMOutput;
  
  // Prints additional output to the terminal during the LMA strategy search if true
  bool verboseLMASearch;
  
  // Determines if the full 3D STEM result is written to file (as individual slices with
  // respect to the detector stepsize
  bool save3DSTEM;
  
  // Detector angles for the 2D STEM output in mrad
  RealType bf_min;
  RealType bf_max;
  
  RealType adf_min;
  RealType adf_max;
  
  RealType haadf_min;
  RealType haadf_max;
  
  
  /** Additional program settings **/
  // Maximum number of threads used
  // 
  // Note: the default value is omp_get_max_threads()
  int max_num_threads;
  
  // Maximum allowed memory consumption in MB
  // 
  // Note: this is a soft threshold. The program tries to stay below this limit, but if at
  //       any point during the initialization or the Multislice computations more memory
  //       has to be used to get correct results, the memory consumption will exceed this
  //       limit
  int memory_limit_MB;
  
  // The higher the value, the more computation time is used to find a
  // good strategy for the Lattice Multislice algorithm. Computation
  // time increases exponentially in the value of this integer.
  int lma_strategy_computation_level;
  
  
  /** Additional settings for recomputing local changes **/
  // Number of additional STEM image simulations for recomputations of
  // local changes.
  // 
  // Note that one additional *.xyz specimen file is required for
  // each additional STEM simulation, in the same directory as the
  // original *.xyz specimen file given by specimen_path and with an
  // integer inserted just before the extension .xyz starting from 1 and
  // up to recomputation_count.
  // Ex.: Original file "SRT.param", additional files "SRT1.param",
  //      "SRT2.param", "SRT3.param" etc.
  // 
  // All other program settings are carried over to the additional
  // simulations, in particular also the tiling parameters tile_x,
  // tile_y and tile_z.
  int recomputation_count;
  
  // Which tile to apply the local changes to (for all simulations).
  // Tile numbering starts from zero.
  int recomputation_change_tile_x;
  int recomputation_change_tile_y;
  int recomputation_change_tile_z;
  
  
  /** Filename of the parameter file **/
  std::string parameterfile_name;
};

// Calculates the electron wavelength in Angstrom from the accelerating voltage given in
// Volt
RealType getWavelength(const RealType AcceleratingVoltage) {
  const RealType c = 299792458;           // Speed of light in meters per second
  const RealType e = 1.60217662e-19;      // Proton charge in Coulomb
  const RealType me = 9.10938356e-31;     // Electron rest mass in kg
  const RealType h = 6.626070040e-34;     // Planck constant in Joule times seconds
  
  const RealType v1 = h / std::sqrt(2*me*AcceleratingVoltage);
  const RealType v2 = 1 / std::sqrt(e);
  const RealType v3 = 1 / std::sqrt(1 + e*AcceleratingVoltage / (2*me*c*c));
  
  return 1e10*v1*v2*v3;
}

// Extract all parameters from a single string
void parseParameterString(const std::string& pstr,
                          Param& p,
                          bool require_necessary_params = true) {
  // Whitespace characters
  const std::string whitespace = " \n\t\f\v\r";
  
  // Extract name-value pairs from pstr
  std::vector<std::pair<std::string, std::string>> name_value;
  
  std::size_t pos = 0;
  while (pos < pstr.length()) {
    // Extract next parameter name (skipping the rest of the line if a # is found)
    pos = pstr.find_first_not_of(whitespace, pos);
    if (pos == std::string::npos)
      break;
    
    if (pstr[pos] == '#') {
      pos = pstr.find_first_of('\n', pos);
      if (pos == std::string::npos)
        break;
      pos = pstr.find_first_not_of(whitespace, pos);
      if (pos == std::string::npos)
        break;
      continue;
    }
    
    std::size_t pos2 = pstr.find_first_of(whitespace, pos);
    if (pos2 == std::string::npos)
      Error("Missing value for parameter \"" + pstr.substr(pos) + "\"!");
    
    std::string pname = pstr.substr(pos, pos2-pos);
    
    // Extract associated parameter value
    pos = pstr.find_first_not_of(whitespace, pos2);
    if (pos == std::string::npos)
      Error("Missing value for parameter \"" + pname + "\"!");
    
    pos2 = pstr.find_first_of(whitespace, pos);
    
    std::string pvalue = pstr.substr(pos, pos2-pos);
    
    // Add to the list of name-value pairs
    name_value.push_back(std::make_pair(pname, pvalue));
    
    pos = pos2;
  }
  
  // Copy the parameter values to p
  auto update_real = [&](const std::string& name,
                         RealType& value,
                         const bool optional_param = false,
                         const RealType default_value = 0) {
    i_update_real(name, value, optional_param, default_value, name_value, require_necessary_params);
  };
  
  auto update_int = [&](const std::string& name,
                        int& value,
                        const bool optional_param = false,
                        const int default_value = 0) {
    i_update_int(name, value, optional_param, default_value, name_value, require_necessary_params);
  };
  
  auto update_bool = [&](const std::string& name,
                         bool& value,
                         const bool optional_param = false,
                         const bool default_value = false) {
    i_update_bool(name, value, optional_param, default_value, name_value, require_necessary_params);
  };
  
  auto update_string = [&](const std::string& name,
                           std::string& value,
                           const bool optional_param = false,
                           const std::string default_value = "") {
    return i_update_string(name, value, optional_param, default_value, name_value, require_necessary_params);
  };
  
  /** Microscope and probe settings **/
  update_real("AcceleratingVoltage", p.AcceleratingVoltage);
  p.lambda = getWavelength(p.AcceleratingVoltage);
  update_real("Z", p.Z);
  update_real("Cs", p.Cs);
  update_real("alpha_max", p.alpha_max);
  
  /** Specimen settings **/
  update_string("specimen_path", p.specimen_path);
  update_int("tile_x", p.tile_x, true, 1);
  update_int("tile_y", p.tile_y, true, 1);
  update_int("tile_z", p.tile_z, true, 1);
  
  /** Simulation mode **/
  std::string tmp;
  
  if (update_string("device", tmp, true, "CPU")) {
    if (tmp == "CPU")
      p.device = MS_Device::CPU;
    else if (tmp == "GPU")
      p.device = MS_Device::GPU;
    else
      Error("Invalid device specified! (Available values are \"CPU\" and \"GPU\")");
  }
  
  if (update_string("domain", tmp, true, "FourierSpace")) {
    if (tmp == "RealSpace")
      p.domain = MS_Domain::RealSpace;
    else if (tmp == "FourierSpace")
      p.domain = MS_Domain::FourierSpace;
    else
      Error("Invalid domain specified! (Available values are \"RealSpace\" and "
                                        "\"FourierSpace\")");
  }
  
  if (update_string("propagator", tmp, true, "Full")) {
    if (tmp == "Full")
      p.propagator = MS_Propagator::Full;
    else if (tmp == "LowRankApproximation")
      p.propagator = MS_Propagator::LowRankApproximation;
    else
      Error("Invalid propagator type specified! (Available values are \"Full\" and "
                                                 "\"LowRankApproximation\")");
  }
  
  if (update_string("inputwave", tmp, true, "Probe")) {
    if (tmp == "Probe")
      p.inputwave = MS_InputWave::Probe;
    else if (tmp == "ProbeSubset")
      p.inputwave = MS_InputWave::ProbeSubset;
    else if (tmp == "TrigonometricPolynomial")
      p.inputwave = MS_InputWave::TrigonometricPolynomial;
    else if (tmp == "TrigonometricPolynomialV2")
      p.inputwave = MS_InputWave::TrigonometricPolynomialV2;
    else if (tmp == "Square")
      p.inputwave = MS_InputWave::Square;
    else if (tmp == "Disk")
      p.inputwave = MS_InputWave::Disk;
    else if (tmp == "Pixel")
      p.inputwave = MS_InputWave::Pixel;
    else if (tmp == "Gaussian")
      p.inputwave = MS_InputWave::Gaussian;
    else if (tmp == "FourierSpaceDirac")
      p.inputwave = MS_InputWave::FourierSpaceDirac;
    else
      Error("Invalid type of inputwave specified! (Available values are "
                                                   "\"Probe\", "
                                                   "\"ProbeSubset\", "
                                                   "\"TrigonometricPolynomial\", "
                                                   "\"TrigonometricPolynomialV2\", "
                                                   "\"Square\", "
                                                   "\"Disk\", "
                                                   "\"Pixel\", "
                                                   "\"Gaussian\" and "
                                                   "\"FourierSpaceDirac\")");
  }
  
  update_int("frozen_phonon_iterations", p.frozen_phonon_iterations, true, 1);
  
  /** Multislice settings **/
  update_real("req_simulation_pixel_size_x", p.req_simulation_pixel_size_x);
  update_real("req_simulation_pixel_size_y", p.req_simulation_pixel_size_y);
  update_real("probe_radius", p.probe_radius);
  update_real("propagation_window_size", p.propagation_window_size, true, 0);
  update_int("propagation_window_test", p.propagation_window_test, true, 5);
  update_real("slice_thickness", p.slice_thickness);
  update_real("potential_bound", p.potential_bound, true, 0.5);
  update_real("detector_stepsize", p.detector_stepsize, true, 1);
  update_int("hires_supersampling_factor", p.hires_supersampling_factor, true, 2);
  update_int("propagator_rank", p.propagator_rank, true, 99999);
  update_real("min_propagator_radius", p.min_propagator_radius, true, 1);
  update_real("fs_buffer_zone", p.fs_buffer_zone, true, -1);
  
  /** Simulation window settings **/
  update_real("req_probe_step_x", p.req_probe_step_x);
  update_real("req_probe_step_y", p.req_probe_step_y);
  update_real("simulation_bounds_x0", p.simulation_bounds_x[0], true, 0);
  update_real("simulation_bounds_x1", p.simulation_bounds_x[1], true, 1);
  update_real("simulation_bounds_y0", p.simulation_bounds_y[0], true, 0);
  update_real("simulation_bounds_y1", p.simulation_bounds_y[1], true, 1);
  
  /** Input wave settings **/
  update_real("max_probe_approximation_error", p.max_probe_approximation_error, true, 0.05);
  update_int("approximation_max_input_waves", p.approximation_max_input_waves, true, -1);
  update_int("approximation_downsampling_factor", p.approximation_downsampling_factor, true, 2);
  update_int("input_wave_lattice_r", p.input_wave_lattice_r, true, -1);
  update_int("trig_poly_degree", p.trig_poly_degree, true, -1);
  update_real("input_wave_square_len", p.input_wave_square_len, true, 2);
  update_real("input_wave_disk_radius", p.input_wave_disk_radius, true, 1);
  update_real("input_wave_pixel_propagation_window_size", p.input_wave_pixel_propagation_window_size, true, 1);
  update_real("input_wave_gaussian_sigma", p.input_wave_gaussian_sigma, true, 0);
  update_bool("input_wave_fsdirac_use_ideal_frequencies", p.input_wave_fsdirac_use_ideal_frequencies, true, true);
  update_int("input_wave_fsdirac_px_beyond_aperture", p.input_wave_fsdirac_px_beyond_aperture, true, 0);
  update_int("test_max_num_input_waves_start", p.test_max_num_input_waves_start, true, -1);
  update_int("test_max_num_input_waves_decrement", p.test_max_num_input_waves_decrement, true, -1);
  
  /** Output settings **/
  update_string("outputDir", p.outputDir, true, "Output");
  update_bool("writeInitData", p.writeInitData, true, true);
  update_bool("writeSlices", p.writeSlices, true, false);
  update_bool("writeProbeApproximationOutput", p.writeProbeApproximationOutput, true, false);
  update_bool("writeLMAStrategySearchOutput", p.writeLMAStrategySearchOutput, true, false);
  update_bool("writePartialSTEMOutput", p.writePartialSTEMOutput, true, true);
  update_bool("verboseLMASearch", p.verboseLMASearch, true, false);
  update_bool("save3DSTEM", p.save3DSTEM, true, true);
  update_real("bf_min", p.bf_min);
  update_real("bf_max", p.bf_max);
  update_real("adf_min", p.adf_min);
  update_real("adf_max", p.adf_max);
  update_real("haadf_min", p.haadf_min);
  update_real("haadf_max", p.haadf_max);
  
  /** Additional program settings **/
  update_int("max_num_threads", p.max_num_threads, true, omp_get_max_threads());
  update_int("memory_limit_MB", p.memory_limit_MB);
  update_int("lma_strategy_computation_level", p.lma_strategy_computation_level, true, 2);
  
  /** Additional settings for recomputing local changes **/
  update_int("recomputation_count", p.recomputation_count, true, 0);
  update_int("recomputation_change_tile_x", p.recomputation_change_tile_x, true, 0);
  update_int("recomputation_change_tile_y", p.recomputation_change_tile_y, true, 0);
  update_int("recomputation_change_tile_z", p.recomputation_change_tile_z, true, 0);
}

// Read all settings and parameters from file
Param readParameterFile(const std::string& filepath) {
  std::ifstream pfile(filepath);
  if (!pfile.is_open())
    Error("Unable to open parameter file \"" + filepath + "\"!");
  
  std::stringstream ss;
  ss << pfile.rdbuf();
  pfile.close();
  
  Param p;
  parseParameterString(ss.str(), p, true);
  
  p.parameterfile_name = std::filesystem::path(filepath).filename();
  
  return p;
}

void writeParameterFile(const std::string& dir, const std::string& filename, const Param& p) {
  std::filesystem::create_directories(dir);
  
  std::ofstream pfile(dir + '/' + filename);
  if(!pfile.is_open())
    Error("Unable to write to file \"" + dir + '/' + filename + "\"!");
  
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  
  pfile << "# Date and time (day/month/year): " << std::put_time(&tm, "%d/%m/%Y  %H:%M:%S") << std::endl
        << std::endl;
  
  pfile << "# Parameter file: \"" << p.parameterfile_name << "\"" << std::endl
        << std::endl;
  
  pfile << "# Microscope and probe settings" << std::endl
        << "AcceleratingVoltage " << p.AcceleratingVoltage << std::endl
        << "# lambda " << p.lambda << std::endl
        << "Z " << p.Z << std::endl
        << "Cs " << p.Cs << std::endl
        << "alpha_max " << p.alpha_max << std::endl
        << std::endl;
  
  pfile << "# Specimen settings" << std::endl
        << "specimen_path " << p.specimen_path << std::endl
        << "tile_x " << p.tile_x << std::endl
        << "tile_y " << p.tile_y << std::endl
        << "tile_z " << p.tile_z << std::endl
        << std::endl;
  
  pfile << "# Simulation mode" << std::endl
        << "device " 
           << (p.device == MS_Device::CPU ? "CPU" :
              (p.device == MS_Device::GPU ? "GPU" :
                                            "ERROR"))
           << std::endl
        << "domain "
           << (p.domain == MS_Domain::RealSpace ?    "RealSpace" :
              (p.domain == MS_Domain::FourierSpace ? "FourierSpace" :
                                                     "ERROR"))
           << std::endl
        << "propagator "
           << (p.propagator == MS_Propagator::Full ?                 "Full" :
              (p.propagator == MS_Propagator::LowRankApproximation ? "LowRankApproximation" :
                                                                     "ERROR"))
           << std::endl
        << "inputwave "
           << (p.inputwave == MS_InputWave::Probe ?                     "Probe" :
              (p.inputwave == MS_InputWave::ProbeSubset ?               "ProbeSubset" :
              (p.inputwave == MS_InputWave::TrigonometricPolynomial ?   "TrigonometricPolynomial" :
              (p.inputwave == MS_InputWave::TrigonometricPolynomialV2 ? "TrigonometricPolynomialV2" :
              (p.inputwave == MS_InputWave::Square ?                    "Square" :
              (p.inputwave == MS_InputWave::Disk ?                      "Disk" :
              (p.inputwave == MS_InputWave::Pixel ?                     "Pixel" :
              (p.inputwave == MS_InputWave::Gaussian ?                  "Gaussian" :
              (p.inputwave == MS_InputWave::FourierSpaceDirac ?         "FourierSpaceDirac" :
                                                                        "ERROR")))))))))
           << std::endl
        << "frozen_phonon_iterations " << p.frozen_phonon_iterations << std::endl
        << std::endl;
  
  pfile << "# Multislice settings" << std::endl
        << "req_simulation_pixel_size_x " << p.req_simulation_pixel_size_x << std::endl
        << "req_simulation_pixel_size_y " << p.req_simulation_pixel_size_y << std::endl
        << "probe_radius " << p.probe_radius << std::endl
        << "propagation_window_size " << p.propagation_window_size << std::endl
        << "propagation_window_test " << p.propagation_window_test << std::endl
        << "slice_thickness " << p.slice_thickness << std::endl
        << "potential_bound " << p.potential_bound << std::endl
        << "detector_stepsize " << p.detector_stepsize << std::endl
        << "hires_supersampling_factor " << p.hires_supersampling_factor << std::endl
        << "propagator_rank " << p.propagator_rank << std::endl
        << "min_propagator_radius " << p.min_propagator_radius << std::endl
        << "fs_buffer_zone " << p.fs_buffer_zone << std::endl
        << std::endl;
  
  pfile << "# Simulation window settings" << std::endl
        << "req_probe_step_x " << p.req_probe_step_x << std::endl
        << "req_probe_step_y " << p.req_probe_step_y << std::endl
        << "simulation_bounds_x0 " << p.simulation_bounds_x[0] << std::endl
        << "simulation_bounds_x1 " << p.simulation_bounds_x[1] << std::endl
        << "simulation_bounds_y0 " << p.simulation_bounds_y[0] << std::endl
        << "simulation_bounds_y1 " << p.simulation_bounds_y[1] << std::endl
        << std::endl;
  
  pfile << "# Input wave settings" << std::endl
        << "max_probe_approximation_error " << p.max_probe_approximation_error << std::endl
        << "approximation_max_input_waves " << p.approximation_max_input_waves << std::endl
        << "approximation_downsampling_factor " << p.approximation_downsampling_factor << std::endl
        << "input_wave_lattice_r " << p.input_wave_lattice_r << std::endl
        << "trig_poly_degree " << p.trig_poly_degree << std::endl
        << "input_wave_square_len " << p.input_wave_square_len << std::endl
        << "input_wave_disk_radius " << p.input_wave_disk_radius << std::endl
        << "input_wave_pixel_propagation_window_size " << p.input_wave_pixel_propagation_window_size << std::endl
        << "input_wave_gaussian_sigma " << p.input_wave_gaussian_sigma << std::endl
        << "input_wave_fsdirac_use_ideal_frequencies " << (p.input_wave_fsdirac_use_ideal_frequencies ? "true" : "false") << std::endl
        << "input_wave_fsdirac_px_beyond_aperture " << p.input_wave_fsdirac_px_beyond_aperture << std::endl
        << "test_max_num_input_waves_start " << p.test_max_num_input_waves_start << std::endl
        << "test_max_num_input_waves_decrement " << p.test_max_num_input_waves_decrement << std::endl
        << std::endl;
  
  pfile << "# Output settings" << std::endl
        << "outputDir " << p.outputDir << std::endl
        << "writeInitData " << (p.writeInitData ? "true" : "false") << std::endl
        << "writeSlices " << (p.writeSlices ? "true" : "false") << std::endl
        << "writeProbeApproximationOutput " << (p.writeProbeApproximationOutput ? "true" : "false") << std::endl
        << "writeLMAStrategySearchOutput " << (p.writeLMAStrategySearchOutput ? "true" : "false") << std::endl
        << "writePartialSTEMOutput " << (p.writePartialSTEMOutput ? "true" : "false") << std::endl
        << "verboseLMASearch " << (p.verboseLMASearch ? "true" : "false") << std::endl
        << "save3DSTEM " << (p.save3DSTEM ? "true" : "false") << std::endl
        << "bf_min " << p.bf_min << std::endl
        << "bf_max " << p.bf_max << std::endl
        << "adf_min " << p.adf_min << std::endl
        << "adf_max " << p.adf_max << std::endl
        << "haadf_min " << p.haadf_min << std::endl
        << "haadf_max " << p.haadf_max << std::endl
        << std::endl;
  
  pfile << "# Additional program settings" << std::endl
        << "max_num_threads " << p.max_num_threads << std::endl
        << "memory_limit_MB " << p.memory_limit_MB << std::endl
        << "lma_strategy_computation_level " << p.lma_strategy_computation_level << std::endl
        << std::endl;
  
  pfile << "# Additional settings for recomputing local changes" << std::endl
        << "recomputation_count " << p.recomputation_count << std::endl
        << "recomputation_change_tile_x " << p.recomputation_change_tile_x << std::endl
        << "recomputation_change_tile_y " << p.recomputation_change_tile_y << std::endl
        << "recomputation_change_tile_z " << p.recomputation_change_tile_z << std::endl
        << std::endl;
}

#endif  // multislice_utility_param_h
