#ifndef multislice_latticemultislicealgorithm_h
#define multislice_latticemultislicealgorithm_h

#include "LatticeMultisliceAlgorithm_aux.h"

#include "LMAStrategy.h"
#include "OutputWindow.h"

#include "../CPU/MSData.h"
#include "../CPU/FFTWData.h"
#include "../CPU/MultisliceAlgorithmCPU.h"

#include "../Utility/Image.h"
#include "../Utility/Image3D.h"
#include "../Utility/Lattice.h"
#include "../Utility/Error.h"

#include <omp.h>

#include <vector>
#include <algorithm>
#include <array>
#include <iterator>

class LatticeMultisliceAlgorithm {
  private:
    // Precomputations for the Multislice algorithm
    mutable MSData data;
    
    // FFTW plans and arrays for each thread
    mutable std::vector<FFTWData> fftw_data;
    
    // Lattice of all probe wave positions over the entire specimen area
    Lattice probe_lattice;
    
    // Lattice of all input wave positions over the entire specimen area
    Lattice input_wave_lattice;
    
    // "Lattice" of exit waves: every entry MultisliceResult(a, b) either stores the
    // result of a Multislice computation corresponding to an input wave centered at
    // position (a, b) of input_wave_lattice or an empty Image of size 0 x 0
    mutable Image<Image<complex>> MultisliceResult;
    
    // Stores the ratio of the norm of the exit wave to the norm of the input wave for all
    // input waves that are transmitted through the specimen, given by their index (a, b)
    // in the input wave lattice
    mutable Image<RealType> norm_change;
    
    // Sets of coefficients for the approximation of the probes by linear combinations of
    // the input waves.
    // 
    // If A * B is the number of input waves over the whole specimen area and X * Y is the 
    // number of probe positions (also over the whole specimen area), then we require the
    // input and probe wave lattices to be compatible, that is, that there are integers
    // c, d >= 1 such that
    //
    //   (Case 1) A * B = (c * X) * (d * Y)      [c = A / X, d = B / Y]
    //
    //            This means that there are more input waves than probe positions and only
    //            one set of coefficients is necessary, i.e. CoefficientSets.size() = 1.
    //            The formula for calculating one Multislice solution for a probe from
    //            known solutions for the input waves is
    //            
    //              sum_{a, b} CoefficientSets(0, 0)(a, b) * MultisliceResult(a, b)
    //              = Multislice solution for the probe at position (x, y) = (0, 0)
    //            
    //            or, more generally, for an arbitrary probe position (x, y),
    //            
    //              sum_{a, b} CoefficientSets(0, 0)(a, b) * MultisliceResult(a + c*x, b + d*y)
    //              = Multislice solution for the probe at position (x, y).
    // 
    // or
    // 
    //   (Case 2) X * Y = (c * A) * (d * B)     [c = X / A, d = Y / B]
    //
    //            In this case there are more probe positions than input waves and c * d
    //            sets of coefficients are necessary for the c * d different probe
    //            approximations from input waves, resulting in CoefficientSets.getX() = c
    //            and CoefficientSets.getY() = d.
    //            The formula for calculating one Multislice solution for a probe from
    //            known solutions for the input waves is then
    //            
    //              sum_{a, b} CoefficientSets(i, j)(a, b) * MultisliceResult(a, b)
    //              = Multislice solution for probe at position (x, y) = (i, j), where
    //                0 <= i < c and 0 <= j < d,
    //
    //            or, more generally, for an arbitrary probe position (x, y),
    //            
    //               sum_{a, b} CoefficientSets(i, j)(a, b) * MultisliceResult(a + floor(x/c), b + floor(y/d))
    //               = Multislice solution for the probe at position (x, y), where
    //                 i = x % CoefficientSets.getX() and j = y % CoefficientSets.getY().
    //
    // Let
    //
    //   c1 = max(1, A / X)  and  d1 = max(1, B / Y),
    //
    //   c2 = max(1, X / A)  and  d2 = max(1, Y / B).
    //
    // Then a formula encompassing both cases for an arbitrary probe position is
    //
    //   sum_{a, b} CoefficientSets(i, j)(a, b) * MultisliceResult(a + floor(x/c2)*c1, b + floor(y/d2)*d1)
    //   = Multislice solution for the probe at position (x, y),
    //
    // where i = x % CoefficientSets.getX() and j = y % CoefficientSets.getY().
    Image<Image<complex>> CoefficientSets;
    
    // Sets of indices of the coefficients that are actually used for the probe
    // approximation, one for each of the CoefficientSets, i.e.
    //
    //   CoefficientIndices.getX() == CoefficientSets.getX() and
    //   CoefficientIndices.getY() == CoefficientSets.getY()
    //
    // The sums over a, b in the description of CoefficientSets above thus do not go over
    // all values of a and b, but only over those in CoefficientIndices(i, j)
    Image<std::vector<std::array<int, 2>>> CoefficientIndices;
    
    // Maximum number of multislice results to store at any given time (to limit the
    // memory consumption)
    int result_storage_limit;
    
    // Dimensions of the resulting 3D STEM image
    OutputWindow output_window;
    
    // Strategy for the Lattice Multislice Algorithm (= partition of the
    // output window into subsets that are computed one at a time)
    LMAStrategy lma_strategy;
    
    // Resulting 3D STEM image
    // Note: only available after calling Compute3DSTEM
    mutable Image3D<RealType> result;
    
    Image<complex> CalculateLinearCombination(const int x, const int y) const;
    std::vector<RealType> Calculate3DSTEMPixel(const int x, const int y) const;
    void CalculateMultisliceResults(const std::vector<std::array<int, 2>> coord) const;
  
  public:
    LatticeMultisliceAlgorithm(const Param& p, const bool compatible_probe_lattice = false);
    
    ~LatticeMultisliceAlgorithm() {
	  for (auto& a : fftw_data)
		a.cleanup();
    }
    
    // Calculate the 3D STEM image for the probe positions given in output_window
    void Compute3DSTEM() const;
    
    // Save 3D STEM image
    void save3D(const std::string& dir) const {
      if (result.empty())
		Error("Called save3D() before Compute3DSTEM()!", __FILE__, __LINE__);
      
      save(result, "slice", dir);
    }
    
    // Reset all Multislice results to free computer memory
    void clearMultisliceResults() const {
      for (int y=0; y<MultisliceResult.getY(); y++)
        for (int x=0; x<MultisliceResult.getX(); x++)
          MultisliceResult(x, y) = Image<complex>();
    }
    
    // Save bright field (BF), annular dark field (ADF) and high angle
    // annular dark field (HAADF) images
    //
    // If a mask is provided, it must be of the same size as the integrated
    // images (output_window.X x output_window.Y). All pixels in the
    // integrated images corresponding to zero values in the mask are set
    // to the average value of the non-zero pixels.
    void saveIntegrated2DImages(const std::string& dir,
                                const Image<RealType>& mask = Image<RealType>(),
                                const int frozen_phonon_iteration = 1) const {
      if (result.empty())
		Error("Called saveIntegrated2DImages() before Compute3DSTEM()!", __FILE__, __LINE__);
      
      if (!mask.empty() && (mask.getX() != output_window.X || mask.getY() != output_window.Y))
        Error("Invalid mask size in saveIntegrated2DImages()!", __FILE__, __LINE__);
      
      std::vector<std::array<RealType, 2>> angles = {{data.p.bf_min, data.p.bf_max},
                                                     {data.p.adf_min, data.p.adf_max},
                                                     {data.p.haadf_min, data.p.haadf_max}};
      std::vector<std::string> names = {"BF", "ADF", "HAADF"};
      
      for (int i=0; i<3; i++) {
		Image<RealType> img = getIntegrated2DImage(result,
                                                   angles[i][0] / data.p.detector_stepsize,
		                                           angles[i][1] / data.p.detector_stepsize);
		
        if (frozen_phonon_iteration != 1)
          img /= static_cast<RealType>(frozen_phonon_iteration);
        
		if (!mask.empty()) {
          // Calculate the average value of pixels corresponding to
          // non-zero mask pixels
	      RealType average = 0;
		  int count = 0;
		  for (int p=0; p<img.size(); p++) {
            if (mask[p] != 0) {
              average += img[p];
              ++count;
            }
          }
          average /= count;
          
          // Set all pixels corresponding to zero mask pixels to the
          // computed average value
          for (int p=0; p<img.size(); p++) {
            if (mask[p] == 0)
              img[p] = average;
          }
        }
        
        save(img, names[i], dir);
      }
    }
    
    // Save an image of the input wave norm changes in the Multislice algorithm
    void saveNormChange(const std::string& dir) const {
      if (result.empty())
		Error("Called saveNormChange() before Compute3DSTEM()!", __FILE__, __LINE__);
      
      save(norm_change, "norms", dir);
    }
    
    // Updates the probe approximation with a new maximum number of input
    // waves used to approximate a single probe, leaving everything else
    // unchanged (in particular the LMA strategy)
    //
    // Returns the approximation errors on the full size (i.e. non downsampled)
    // images in the relative euclidean and supremum norms, as an image of the
    // same size as CoefficientSets or CoefficientIndices (i.e. one approximation
    // error for each approximation).
    void updateProbeApproximation(const int max_num_input_waves,
                                  Image<RealType>& approximation_errors_euc,
                                  Image<RealType>& approximation_errors_sup) {
      initProbeApproximation(data,
                             probe_lattice,
                             input_wave_lattice,
                             CoefficientSets,
                             CoefficientIndices,
                             max_num_input_waves,
                             false,
                             &approximation_errors_euc,
                             &approximation_errors_sup,
                             true);
    }
    
    // Calculates the relative error of other.result as compared to
    // this->result, which is considered as the ground truth.
    void calculateRelativeError(const LatticeMultisliceAlgorithm& other,
                                RealType& diff3d_euc,
                                RealType& diff3d_sup,
                                std::array<RealType, 3>& diff2d_euc,
                                std::array<RealType, 3>& diff2d_sup,
                                std::array<Image<RealType>, 3>& diff2d_images) const {
      if (result.empty() || other.result.empty())
        Error("Called calculateDifference() before Compute3DSTEM()!", __FILE__, __LINE__);
      
      diff3d_euc = euc_dist(result, other.result) / euc_norm(result);
      diff3d_sup = sup_dist(result, other.result) / sup_norm(result);
      
      std::vector<std::array<RealType, 2>> angles = {{data.p.bf_min, data.p.bf_max},
                                                     {data.p.adf_min, data.p.adf_max},
                                                     {data.p.haadf_min, data.p.haadf_max}};
      
      for (int i=0; i<3; i++) {
        Image<RealType> img = getIntegrated2DImage(result,
                                                   angles[i][0] / data.p.detector_stepsize,
                                                   angles[i][1] / data.p.detector_stepsize);
        
        diff2d_images[i] = getIntegrated2DImage(other.result,
                                                angles[i][0] / data.p.detector_stepsize,
                                                angles[i][1] / data.p.detector_stepsize);
        
        diff2d_images[i] -= img;
        
        const RealType sup_norm_img = sup_norm(img);
        
        diff2d_euc[i] = norm(diff2d_images[i]) / norm(img);
        diff2d_sup[i] = sup_norm(diff2d_images[i]) / sup_norm_img;
        
        for (int j=0; j<diff2d_images[i].size(); j++)
          diff2d_images[i][j] = std::abs(diff2d_images[i][j]) / sup_norm_img;
      }
    }
    
    // Perform only the necessary recomputations to calculate a 3D STEM
    // image for the n-th specimen *.xyz file
    void Recompute3DSTEM(const int n) const;
};

LatticeMultisliceAlgorithm::LatticeMultisliceAlgorithm(const Param& p, const bool compatible_probe_lattice) : data(p) {
    if (p.device == MS_Device::GPU)
    Error("Not implemented! (device = GPU)", __FILE__, __LINE__);
  
  // Perform necessary initializations for FFTW if the Fourier space algorithm is used on the CPU
  if (p.device == MS_Device::CPU && p.domain == MS_Domain::FourierSpace) {
    std::cerr << "Initializing FFTW plans using FFTW_PATIENT. This may take some time ..." << std::endl;
    initFFTW(data, fftw_data);
  }
  
  // Initialize the lattices of the input wave positions and the probe positions
  initPositionLattices(data, probe_lattice, input_wave_lattice, compatible_probe_lattice);
  
  // Test the propagation window size if requested
  if (p.propagation_window_test > 0 && p.inputwave != MS_InputWave::FourierSpaceDirac)
    testPropagationWindowSize(data, fftw_data, input_wave_lattice);
  
  // Initialize MultisliceResult and the norm change matrix
  MultisliceResult = Image<Image<complex>>(input_wave_lattice.X, input_wave_lattice.Y, Image<complex>());
  norm_change = Image<RealType>(input_wave_lattice.X, input_wave_lattice.Y, 1);
  
  // Initialize the probe positions and the detector angles (the output window)
  output_window = OutputWindow(p, probe_lattice, data.propagation_window);
  
  // Calculate the coefficient sets and indices for the approximation of the probes
  // by linear combinations of the input waves
  initProbeApproximation(data,
                         probe_lattice,
                         input_wave_lattice,
                         CoefficientSets,
                         CoefficientIndices,
                         data.p.approximation_max_input_waves,
                         true);
  
  // Calculate result_storage_limit from the size of the exit waves, the number of
  // input waves required for the approximation of a single probe position (hard lower
  // bound), and the requested maximum amount of memory consumption with
  // memory_limit_MB (soft upper bound)
  AlignedArray sample_input_window = getInputWaveWindow(data, {0, 0}, {0, 0}, {0, 0});
  int exit_wave_pixel = sample_input_window.aa_size[0] * sample_input_window.aa_size[1];
  
  int num_input_waves = 0;
  for (int k=0; k<CoefficientIndices.size(); k++)
    num_input_waves = std::max(num_input_waves, static_cast<int>(CoefficientIndices[k].size()));
  
  const RealType ew_MB = exit_wave_pixel * sizeof(complex) / static_cast<RealType>(1024 * 1024);
  
  result_storage_limit = static_cast<int>(data.p.memory_limit_MB / ew_MB);
  
  if (result_storage_limit < num_input_waves) {
    std::cerr << "WARNING: the Lattice Multislice Algorithm requires at least " << static_cast<int>(num_input_waves * ew_MB) << " MB" << std::endl
              << "         of computer memory. This exceeds the requested memory limit" << std::endl
              << "         of " << data.p.memory_limit_MB << " MB." << std::endl
              << std::endl;
    
    result_storage_limit = num_input_waves;
  }
  
  // Prepare a strategy for the Lattice Multislice Algorithm
	lma_strategy = chooseLMAStrategy(output_window,
																	 CoefficientIndices,
																	 input_wave_lattice,
																	 probe_lattice,
																	 result_storage_limit,
																	 data.p.lma_strategy_computation_level,
																	 data.p.writeLMAStrategySearchOutput,
																	 data.p.verboseLMASearch,
																	 data.p.outputDir,
																	 p);
  
  // Calculate the probe window size in pixel
  AlignedArray probe_window = getProbeWindow(data, {0, 0});
  
  // Print a brief overview of the configuration
  std::cerr << std::endl
            << "Lattice Multislice Algorithm configuration" << std::endl
            << "----+--------------------------------------------------------------" << std::endl
            << "    | Output window dimensions (3D STEM): " << output_window.X << " x " << output_window.Y << " x " << output_window.Z << " (pixel)" << std::endl
            << "    |" << std::endl
            << "    | Probe lattice" << std::endl
            << "    |----+------------------------------------------------" << std::endl
            << "    |    | Number of probe positions: " << probe_lattice.X << " x " << probe_lattice.Y << std::endl
            << "    |    |            Probe stepsize: (" << probe_lattice.dx << ", " << probe_lattice.dy << ") (in Angstrom)" << std::endl
            << "    |    |              Probe offset: (" << probe_lattice.offset_x << ", " << probe_lattice.offset_y << ") (in Angstrom)" << std::endl
            << "    |----+------------------------------------------------" << std::endl
            << "    |" << std::endl
            << "    | Input wave lattice" << std::endl
            << "    |----+------------------------------------------------" << std::endl
            << "    |    | Number of input wave positions: " << input_wave_lattice.X << " x " << input_wave_lattice.Y << std::endl
            << "    |    |            Input wave stepsize: (" << input_wave_lattice.dx << ", " << input_wave_lattice.dy << ") (in Angstrom)" << std::endl
            << "    |    |              Input wave offset: (" << input_wave_lattice.offset_x << ", " << input_wave_lattice.offset_y << ") (in Angstrom)" << std::endl
            << "    |----+------------------------------------------------" << std::endl
            << "    |" << std::endl
            << "    | Output window partition (LMA strategy)" << std::endl
            << "    |----+------------------------------------------------" << std::endl
            << "    |    |                         Strategy: " << lma_strategy.name << std::endl
            << "    |    |                   Number of sets: " << lma_strategy.computation_domains.size() << std::endl
            << "    |    | Average number of pixels per set: " << static_cast<RealType>(output_window.X * output_window.Y) / lma_strategy.computation_domains.size() << std::endl
            << "    |    | Total number of Multislice calls: " << lma_strategy.num_multislice_computations << std::endl
            << "    |----+------------------------------------------------" << std::endl
            << "    |" << std::endl
            << "    | Number of different sets of coefficients: " << CoefficientSets.getX() << " * " << CoefficientSets.getY() << std::endl
            << "    | Maximum number of input waves used for the approximation of one probe: " << num_input_waves << std::endl
            << "    |" << std::endl
            << "    | Size of the exit waves: " << data.propagation_window.X << " x " << data.propagation_window.Y << " (in pixel)" << std::endl
            << "    | Memory required to store one exit wave: " << ew_MB << " (in MB)" << std::endl
            << "    | Maximum number of exit waves that may be kept in memory at the same time: " << result_storage_limit << std::endl
            << "    |" << std::endl
            << "    | Probe window size: " << probe_window.aa_size[0] << " x " << probe_window.aa_size[1] << " (in pixel)" << std::endl
            << "    | (Remark: this is the size of the result after forming the linear combinations of the exit waves)" << std::endl
            << "----+--------------------------------------------------------------" << std::endl
            << std::endl;
}

// Calculates the solution to the Schrödinger equation for the probe at position
// (x, y) in the probe lattice, assuming (!) that all the required exit waves in
// MultisliceResult have been computed and are stored in MultisliceResults
Image<complex> LatticeMultisliceAlgorithm::CalculateLinearCombination(const int x, const int y) const {
  // Get the probe position in Angstrom
  std::array<RealType, 2> probe_pos = probe_lattice.getPosition(x, y);
  
  // Get the probe window coordinates
  AlignedArray probe_window = getProbeWindow(data, probe_pos);
  
  // The multislice solution for the probe at position (x, y) is given by
  //
  //   sum_{a, b} CoefficientSets(i, j)(a, b) * MultisliceResult(a + floor(x/c2)*c1, b + floor(y/d2)*d1)
  //   = Multislice solution for the probe at position (x, y),
  //
  // where i = x % CoefficientSets.getX() and j = y % CoefficientSets.getY().
  // (See also the explanation of CoefficientSets in the LatticeMultisliceAlgorithm class)
  const int i = x % CoefficientSets.getX();
  const int j = y % CoefficientSets.getY();
  
  // Calculate the linear combination
  //
  // Note: the exit waves in MultisliceResult are cropped to the probe window
  Image<complex> result(probe_window.aa_size[0], probe_window.aa_size[1], {0, 0});
  for (std::array<int, 2> coeff_coord: CoefficientIndices(i, j)) {
    // Get the input wave coordinates in the input wave lattice
    const std::array<int, 2> iw_coord = getInputWaveCoord({x, y}, coeff_coord, input_wave_lattice, probe_lattice, data.p);
    
    // Get the input wave position in Angstrom
    const std::array<RealType, 2> input_wave_pos = input_wave_lattice.getPosition(iw_coord[0], iw_coord[1]);
    
    // Get the input wave window coordinates
    AlignedArray input_window = getInputWaveWindow(data, input_wave_pos, coeff_coord, probe_pos);
		
    // Calculate the current summand ...
    if (MultisliceResult(iw_coord[0], iw_coord[1]).size() == 0)
      Error("A required multislice solution is not available!", __FILE__, __LINE__);
    
    Image<complex> exit_wave = MultisliceResult(iw_coord[0], iw_coord[1]);
    
    if (data.p.inputwave == MS_InputWave::FourierSpaceDirac) {
			// Modulate the coefficients appropriately
			const RealType fs_pixel_size_x = 1 / (data.simulation_window.pixel_size_x * data.simulation_window.X);
			const RealType fs_pixel_size_y = 1 / (data.simulation_window.pixel_size_y * data.simulation_window.Y);
			
			std::array<RealType, 2> wave_fs_pos = {coeff_coord[0] * fs_pixel_size_x,
				                                     coeff_coord[1] * fs_pixel_size_y};
		  std::array<RealType, 2> neg_probe_rs_pos = {-probe_pos[0],
				                                          -probe_pos[1]};
		  
      exit_wave *= CoefficientSets(i, j)(iw_coord[0], iw_coord[1]) * modulation(wave_fs_pos, neg_probe_rs_pos);
    } else
      exit_wave *= CoefficientSets(i, j)(coeff_coord[0], coeff_coord[1]);
    
    // ... and add it to the result
    if (data.p.inputwave == MS_InputWave::FourierSpaceDirac) {
			result += exit_wave.getPeriodic(probe_window.aa_size[0],
																		  probe_window.aa_size[1],
																		  probe_window.aa_pos[0],
																		  probe_window.aa_pos[1]);
		} else {
			std::array<int, 2> probe_window_start;
			std::array<int, 2> input_window_start;
			std::array<int, 2> intersection_size;
			
			getOverlappingWindowArea(data,
															 probe_window,
															 input_window,
															 probe_window_start,
															 input_window_start,
															 intersection_size);
			
			for (int y=0; y<intersection_size[1]; y++)
				for (int x=0; x<intersection_size[0]; x++)
					 result(probe_window_start[0] + x, probe_window_start[1] + y)
						+= exit_wave(input_window_start[0] + x, input_window_start[1] + y);
		}
  }
  
  return result;
}

// Calculates one pixel value (output_window.start_x + x, output_window.start_y + y) of
// the final 3D STEM image, assuming (!) that the multislice solutions for the required
// input waves have been computed and are stored in MultisliceResult
std::vector<RealType> LatticeMultisliceAlgorithm::Calculate3DSTEMPixel(const int x, const int y) const {
  Image<complex> probe_exit_wave = CalculateLinearCombination(output_window.start_x + x,
                                                              output_window.start_y + y);
  
  // Get the size of the probe window in Angstrom
  const RealType probe_window_lenX = probe_exit_wave.getX() * data.propagation_window.pixel_size_x;
  const RealType probe_window_lenY = probe_exit_wave.getY() * data.propagation_window.pixel_size_y;
  
  std::vector<RealType> detector_intensity = calculateDetectorValues(probe_exit_wave,
                                                                     probe_window_lenX,
                                                                     probe_window_lenY,
                                                                     data.p,
                                                                     output_window.Z);
  
  return detector_intensity;
}

// Calculates the result of transmitting the input wave at the positions iw_coords[i] of
// the input wave lattice and stores the results in
//
//     MultisliceResult(iw_coords[i][0], iw_coords[i][1])
//
void LatticeMultisliceAlgorithm::CalculateMultisliceResults(const std::vector<std::array<int, 2>> iw_coords) const {
  std::cerr << "Performing " << iw_coords.size() << " multislice calculations ..." << std::endl;
  
  if (data.p.device == MS_Device::CPU) {
    int progress = 0;
    
    #pragma omp parallel for num_threads(data.p.max_num_threads)
    for (unsigned int i=0; i<iw_coords.size(); i++) {
      const int x = iw_coords[i][0];
      const int y = iw_coords[i][1];
      
      std::array<RealType, 2> pos = input_wave_lattice.getPosition(x, y);
      const AlignedArray input_wave_window = getInputWaveWindow(data,
                                                                pos,
                                                                {x - input_wave_lattice.X/2, y - input_wave_lattice.Y/2},
                                                                {0, 0} /* irrelevant */);
      
      MultisliceResult(x, y) = MultisliceAlgorithmCPU(input_wave_window,
                                                      fftw_data[omp_get_thread_num()],
                                                      data,
                                                      norm_change(x, y));
      
      #pragma omp critical (CalculateMultisliceResults_CPU)
      {
        std::cerr << "\r\tProgress: " << ++progress << " / " << iw_coords.size();
      }
    }
    std::cerr << std::endl;
    
  } else if (data.p.device == MS_Device::GPU) {
    Error("Not implemented! (device = GPU)", __FILE__, __LINE__);
  } else
    Error("Not implemented!", __FILE__, __LINE__);
}

// Calculate a 3D STEM image for the probe positions given in output_window
void LatticeMultisliceAlgorithm::Compute3DSTEM() const {
  std::cerr << "Computing STEM image ..." << std::endl
            << std::endl;
  
  // Reset the output image and the norm change array
  result = Image3D<RealType>(output_window.X, output_window.Y, output_window.Z);
  norm_change = Image<RealType>(input_wave_lattice.X, input_wave_lattice.Y, 1);
  
  for (int fp=0; fp<data.p.frozen_phonon_iterations; fp++) {
    int current_step = 0;
    const int num_steps = static_cast<int>(lma_strategy.computation_domains.size());
    
    std::vector<std::array<int, 2>> cur_iw_pos, prev_iw_pos;
    
    for (const auto& cd: lma_strategy.computation_domains) {
      std::cerr << "---------------------------------------------------" << std::endl;
      if (data.p.frozen_phonon_iterations != 1)
        std::cerr << "Frozen phonon iteration " << fp+1 << " of " << data.p.frozen_phonon_iterations << std::endl;
      std::cerr << "Subset " << ++current_step << " of " << num_steps << " (" << cd.size() << " pixel)" << std::endl << std::endl;
      
      cur_iw_pos = getRequiredInputWavePositions(cd,
                                                 CoefficientIndices,
                                                 input_wave_lattice,
                                                 probe_lattice,
                                                 output_window.start_x,
                                                 output_window.start_y,
                                                 data.p);
      
      // Delete all Multislice results for input wave positions that are in
      // prev_iw_pos but not in cur_iw_pos
      auto cmp = [&](const std::array<int, 2>& a, const std::array<int, 2>& b) {
        return (a[0] + a[1] * input_wave_lattice.X < b[0] + b[1] * input_wave_lattice.X);
      };
      
      std::vector<std::array<int, 2>> to_delete;
      std::set_difference(prev_iw_pos.begin(), prev_iw_pos.end(),
                          cur_iw_pos.begin(), cur_iw_pos.end(),
                          std::inserter(to_delete, to_delete.begin()),
                          cmp);
      
      for (const std::array<int, 2>& index: to_delete)
        MultisliceResult(index[0], index[1]) = Image<complex>();
      
      // Calculate all multislice results for input wave positions that are in cur_iw_pos
      // but not in prev_iw_pos
      std::cerr << "Step 1: Precomputing multislice solutions ..." << std::endl;
      
      std::vector<std::array<int, 2>> to_compute;
      std::set_difference(cur_iw_pos.begin(), cur_iw_pos.end(),
                          prev_iw_pos.begin(), prev_iw_pos.end(),
                          std::inserter(to_compute, to_compute.begin()),
                          cmp);
      
      CalculateMultisliceResults(to_compute);
      
      // Form linear combinations to get the result for the probe wave functions
      std::cerr << "Step 2: Calculating linear combinations ..." << std::endl;
      
      int progress = 0;
      #pragma omp parallel for num_threads(data.p.max_num_threads)
      for (unsigned int i=0; i<cd.size(); i++) {
        const auto& p = cd[i];  // Some implementations of omp don't like for-each loops
        std::vector<RealType> pixel = Calculate3DSTEMPixel(p[0], p[1]);
        
        for (int z=0; z<output_window.Z; z++)
          result(p[0], p[1], z) += pixel[z];
        
        #pragma omp critical (Compute3DSTEM_linearcombinations)
        {
          std::cerr << "\r\tProgress: " << ++progress << " / " << cd.size();
        }
      }
      std::cerr << std::endl << std::endl;
      
      // Save partial results if requested
      if (data.p.writePartialSTEMOutput) {
        std::cerr << "Saving partial results ...";
        
        Image<RealType> progress_image = lma_strategy.getProgressImage(current_step, output_window);
        
        saveIntegrated2DImages(data.p.outputDir + "/PartialResult/2D", progress_image, fp+1);
        saveNormChange(data.p.outputDir + "/PartialResult/ExitWaveNormChange");
        save(progress_image, "ComputedPixels", data.p.outputDir + "/PartialResult");
        
        std::cerr << std::endl
                  << std::endl;
      }
      
      // Keep track of the input wave positions used for the previous rows
      prev_iw_pos = cur_iw_pos;
    }
    
    // Recompute the transmission functions with randomly displaced atoms
    // according to their vibration factors
    if (fp != data.p.frozen_phonon_iterations-1) {
      std::cerr << "Recomputing the specimen potential for the next frozen phonon iteration ..." << std::endl;
      
      data.nextFrozenPhononState();
    }
  }
  
  // Average the results from the frozen phonon iterations, which have all
  // been added to result
  if (data.p.frozen_phonon_iterations != 1)
    result /= static_cast<RealType>(data.p.frozen_phonon_iterations);
}

// Perform the necessary recomputations for the n-th specimen file
void LatticeMultisliceAlgorithm::Recompute3DSTEM(const int n) const {
  std::cerr << "Recomputing local changes (n = " << n << ") ..." << std::endl
            << std::endl;
  
  if (result.empty())
    Error("Compute3DSTEM() must be called before Recompute3DSTEM()!", __FILE__, __LINE__);
  
  // Update the specimen and the transmission functions
  std::cerr << "Step 1: Updating the specimen structure ..." << std::endl;
  
  std::vector<std::array<RealType, 3>> change_locations = data.updateTransmissionFunctions(n);
  
  if (change_locations.empty()) {
    std::cerr << "\tWARNING: found no changes in the specimen structure." << std::endl;
  } else {
    std::cerr << "\tFound " << change_locations.size() << " locations with changed specimen structure (in Angstrom):" << std::endl;
    
    for (auto p: change_locations)
      std::cerr << "\t\t(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
  
  if (data.p.device == MS_Device::GPU) {
    Error("Not implemented! (device = GPU)", __FILE__, __LINE__);
  }
  
  // Note: we don't (!) reset the MultisliceResult and norm_change matrices
  
  // Calculate a vector of all probe positions that need to be recomputed
  // within the output window
  std::cerr << "Step 2: Calculating pixels that need to be recomputed ..." << std::endl;
  
  const std::vector<std::array<int, 2>> probe_pos = getProbeUpdatePositions(data,
                                                                            change_locations,
                                                                            probe_lattice,
                                                                            output_window);
  
  if (data.p.writeInitData)
    save(getIndicatorImage(probe_pos, output_window.X, output_window.Y),
         "probe_update_positions",
         data.p.outputDir + "/LocalChanges/" + std::to_string(n));
  
  std::cerr << "\t:: " << probe_pos.size() << " pixel need to be recomputed." << std::endl;
  
  // Calculate a vector of all input wave positions for which MultisliceResult
  // needs to be computed or recomputed
  std::cerr << "Step 3: Finding necessary Multislice computations ..." << std::endl;
  
  std::vector<std::array<int, 2>> to_delete;
  std::vector<std::array<int, 2>> to_compute = getInputWaveUpdatePositions(data,
                                                                           probe_pos,
                                                                           change_locations,
                                                                           input_wave_lattice,
                                                                           probe_lattice,
                                                                           output_window,
                                                                           MultisliceResult,
                                                                           CoefficientIndices,
                                                                           to_delete);
  
  if (data.p.writeInitData) {
    save(getIndicatorImage(to_compute, input_wave_lattice.X, input_wave_lattice.Y),
         "input_wave_update_positions",
         data.p.outputDir + "/LocalChanges/" + std::to_string(n));
    
    save(getIndicatorImage(to_delete, input_wave_lattice.X, input_wave_lattice.Y),
         "input_wave_delete_positions",
         data.p.outputDir + "/LocalChanges/" + std::to_string(n));
  }
  
  std::cerr << "\t:: " << to_compute.size() << " Multislice computations need to be performed." << std::endl
            << "\t:: " << to_delete.size() << " old Multislice results are deleted." << std::endl;
  
  // Delete all entries of MultisliceResult that are not required
  for (const std::array<int, 2>& index: to_delete)
    MultisliceResult(index[0], index[1]) = Image<complex>();
  
  // Frozen phonon approximation
  if (data.p.frozen_phonon_iterations != 1)
    std::cerr << std::endl;
  
  Image3D<RealType> recomputed_pixels(output_window.X, output_window.Y, output_window.Z);
  
  for (int fp=0; fp<data.p.frozen_phonon_iterations; fp++) {
    if (data.p.frozen_phonon_iterations != 1)
      std::cerr << "Frozen phonon iteration " << fp+1 << " of " << data.p.frozen_phonon_iterations << std::endl
                << "---------------------------------------------" << std::endl;
    
    // After the first frozen phonon iteration all atoms are randomly displaced,
    // which means that from then onwards all required input wave positions
    // need to be recomputed and not only those close to a change location.
    if (fp == 1)
      to_compute = getRequiredInputWavePositions(probe_pos,
                                                 CoefficientIndices,
                                                 input_wave_lattice,
                                                 probe_lattice,
                                                 output_window.start_x,
                                                 output_window.start_y,
                                                 data.p);
  
    
    // Calculate all required multislice results for the input wave positions in
    // input_waves
    std::cerr << "Step 4: (Re)computing multislice solutions ..." << std::endl;
    
    CalculateMultisliceResults(to_compute);
    
    // Form linear combinations to update the pixels in probe_pos
    std::cerr << "Step 5: Calculating linear combinations ..." << std::endl;
    
    int progress = 0;
    #pragma omp parallel for num_threads(data.p.max_num_threads)
    for (unsigned int i=0; i<probe_pos.size(); i++) {
      const auto& p = probe_pos[i];  // Some implementations of omp don't like for-each loops
      std::vector<RealType> pixel = Calculate3DSTEMPixel(p[0], p[1]);
      
      for (int z=0; z<output_window.Z; z++)
        recomputed_pixels(p[0], p[1], z) += pixel[z];
      
      #pragma omp critical (Recompute3DSTEM_linearcombinations)
      {
        std::cerr << "\r\tProgress: " << ++progress << " / " << probe_pos.size();
      }
    }
    std::cerr << std::endl
              << std::endl;
    
    // Recompute the transmission functions with randomly displaced atoms
    // according to their vibration factors
    if (fp != data.p.frozen_phonon_iterations-1) {
      std::cerr << "Recomputing the specimen potential for the next frozen phonon iteration ..." << std::endl;
      
      data.nextFrozenPhononState();
    }
  }
  
  // Assign the new (averaged) values to result
  for (const auto& p: probe_pos)
    for (int z=0; z<output_window.Z; z++)
      result(p[0], p[1], z) = recomputed_pixels(p[0], p[1], z) / data.p.frozen_phonon_iterations;
}

#endif  // multislice_latticemultislicealgorithm_h
