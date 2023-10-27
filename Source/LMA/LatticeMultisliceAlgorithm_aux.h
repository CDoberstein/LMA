// Auxiliary functions for the LatticeMultisliceAlgorithm class
#ifndef multislice_latticemultislicealgorithm_aux_h
#define multislice_latticemultislicealgorithm_aux_h

#include "OutputWindow.h"
#include "LMAStrategy_aux.h"  // getRequiredInputWavePositions()

#include "../CPU/MSData.h"
#include "../CPU/FFTWData.h"
#include "../CPU/ProbeApproximation.h"
#include "../CPU/MultisliceAlgorithmCPU.h"

#include "../Utility/Image.h"
#include "../Utility/Lattice.h"
#include "../Utility/Specimen.h"
#include "../Utility/Error.h"

#include <omp.h>

#include <vector>
#include <utility>
#include <cmath>
#include <array>
#include <algorithm>
#include <set>

// Allocates FFTW arrays and creates plans for the forward and inverse Fourier
// transforms in MultisliceAlgorithmFS()
void initFFTW(const MSData& data,
              std::vector<FFTWData> &fftw_data) {
  fftw_data.resize(data.p.max_num_threads);
  
  int counter = 0;
  for (auto& a: fftw_data) {
    std::cerr << "\r\t" << counter++ << " / " << data.p.max_num_threads << "     ";
    
    if (data.p.fs_buffer_zone >= 0)
      a.init(data.propagation_window.ext_X, data.propagation_window.ext_Y);
    else
	    a.init(data.simulation_window.X, data.simulation_window.Y);
  }
  std::cerr << "\r\t" << data.p.max_num_threads << " / " << data.p.max_num_threads << "     " << std::endl << std::endl;
}

// Run the Multislice algorithm for a few input wave positions to check
// if the propagation window size is sufficiently large
void testPropagationWindowSize(const MSData& data,
                               std::vector<FFTWData>& fftw_data,
                               const Lattice& input_wave_lattice) {
  std::cerr << "Running the Multislice algorithm " << data.p.propagation_window_test
            << " times to test the size of the propagation window ..." << std::endl;
  
  // Save an image of the input wave
  Image<complex> input_wave = data.input.getInitialCondition(data.propagation_window.pixel_size_x,
                                                             data.propagation_window.pixel_size_y,
                                                             data.propagation_window.X,
                                                             data.propagation_window.Y,
                                                             data.propagation_window.pixel_size_x * (data.propagation_window.X / 2),
                                                             data.propagation_window.pixel_size_y * (data.propagation_window.Y / 2));
  save(input_wave, "InputWave", data.p.outputDir + "/PropagationWindowTest/InputWave");
  
  // Perform Multislice simulations and save the results
  int counter = 0;
  #pragma omp parallel for num_threads(data.p.max_num_threads)
  for (int i=0; i<data.p.propagation_window_test; i++) {
    const int p = static_cast<int>((i * static_cast<RealType>(input_wave_lattice.X * input_wave_lattice.Y)) / data.p.propagation_window_test);
    
    const int x = p % input_wave_lattice.X;
    const int y = p / input_wave_lattice.X;
    
    std::array<RealType, 2> pos = input_wave_lattice.getPosition(x, y);
    const AlignedArray input_wave_window = getInputWaveWindow(data,
                                                              pos,
                                                              {0, 0} /* irrelevant */,
                                                              {0, 0} /* irrelevant */);
    
    RealType norm_change;
    Image<complex> res = MultisliceAlgorithmCPU(input_wave_window,
                                                fftw_data[omp_get_thread_num()],
                                                data,
                                                norm_change);
    
    #pragma omp critical (testPropagationWindowSize)
    {
      ++counter;
      std::cerr << "\tNorm change: " << norm_change << " (" << counter
                << " / " << data.p.propagation_window_test << ")" << std::endl;
      
      save(res,
           "Result_" + std::to_string(x) + "_" + std::to_string(y),
           data.p.outputDir + "/PropagationWindowTest");
    }
  }
  
  std::cerr << "Done. Results can be found in \"" << data.p.outputDir + "/PropagationWindowTest" << "\"." << std::endl
            << std::endl;
}

// Calculates an upper bound for the dimension of the probe vector space by counting
// the number of pixels within the objective aperture in Fourier space
int maxProbeVectorSpaceDimension(const MSData& data) {
  // Aperture radius in 1/Angstrom
  const RealType aperture_radius = data.p.alpha_max / data.p.lambda;
  
  // Fourier space pixel size for the computation of an aperture image
  RealType fs_pixel_size;
  
  if (data.p.domain == MS_Domain::RealSpace)
    fs_pixel_size = 1/data.propagation_window.lenX;
  else if (data.p.domain == MS_Domain::FourierSpace)
    fs_pixel_size = std::min(1/data.simulation_window.lenX, 1/data.simulation_window.lenY);
  else
    Error("Not implemented!", __FILE__, __LINE__);
  
  // Circular image of the aperture in Fourier space
  const int num_points = static_cast<int>(std::ceil(aperture_radius / fs_pixel_size));
  CircularImage<float> aperture(std::vector<float>(num_points, 1.f), fs_pixel_size);
  
  // Convert to an ordinary 2D image and count the pixels within the aperture
  // (= the nonzero pixels)
  Image<float> aperture2D;
  if (data.p.domain == MS_Domain::RealSpace)
    aperture2D = aperture.get2DImage(fs_pixel_size, 0.f);
  else if (data.p.domain == MS_Domain::FourierSpace)
    aperture2D = aperture.get2DImageNonSquare(1/data.simulation_window.lenX,
                                              1/data.simulation_window.lenY,
                                              0.f);
  else
    Error("Not implemented!", __FILE__, __LINE__);
  
  // Return the number of nonzero pixels of aperture2D
  int count = 0;
  for (int i=0; i<aperture2D.size(); i++)
    if (aperture2D[i] != 0.f)
      ++count;
  
  return count;
}

// Calculates the lattices of all input wave positions and all probe wave positions over
// the entire specimen area
//
// If compatible_probe_lattice is true and the input wave type is "Probe",
// then the probe lattice will be initialized as if the input wave type was
// "ProbeSubset" (or one of the other types handled identically)
void initPositionLattices(const MSData& data,
                          Lattice& probe_lattice,
                          Lattice& input_wave_lattice,
                          const bool compatible_probe_lattice) {
  switch (data.p.inputwave) {
    case MS_InputWave::Probe: 
      if (!compatible_probe_lattice) {
        // Standard Multislice algorithm: the input wave lattice and the probe lattice are
        // identical
        
        // Adjust the probe step size (dx, dy) to ensure that the specimen width and height
        // are integer multiples of the probe step size. This is necessary because of the
        // implicit periodic repetition of the specimen
        probe_lattice.X = static_cast<int>(std::ceil(data.specimen.lenX / data.p.req_probe_step_x));
        probe_lattice.Y = static_cast<int>(std::ceil(data.specimen.lenY / data.p.req_probe_step_y));
        
        probe_lattice.dx = data.specimen.lenX / probe_lattice.X;
        probe_lattice.dy = data.specimen.lenY / probe_lattice.Y;
        
        probe_lattice.offset_x = 0;
        probe_lattice.offset_y = 0;
        
        input_wave_lattice = probe_lattice;
        
        break; // This must be inside the true-branch of the above if
      }
    case MS_InputWave::TrigonometricPolynomial:
    case MS_InputWave::TrigonometricPolynomialV2:
    case MS_InputWave::Square:
    case MS_InputWave::Disk:
    case MS_InputWave::Gaussian:
    case MS_InputWave::ProbeSubset: {
      // Modification of the standard Multislice algorithm: the input wave lattice
      // contains less points than the probe lattice if the dimension of the probe vector
      // space is smaller than the number of probe positions in the probe lattice
      int num_probes_x = static_cast<int>(std::ceil(data.specimen.lenX / data.p.req_probe_step_x));
      int num_probes_y = static_cast<int>(std::ceil(data.specimen.lenY / data.p.req_probe_step_y));
      
      const int max_dim = maxProbeVectorSpaceDimension(data);
      const int requested_num_probes = num_probes_x * num_probes_y;
      
      // Factor by which the number of probes in both x and y direction is reduced:
      // Instead of propagating requested_num_probes probes through the specimen, only
      // requested_num_probes / r^2 input wave positions are considered.
      //
      // Note: by the choice of r, requested_num_probes / r^2 >= max_dim unless
      //       max_dim is greater than requested_num_probes or p.input_wave_lattice_r is
      //       manually set to a value that does not satisfy this inequality.
      const int r = ( data.p.input_wave_lattice_r > 0
                      ? data.p.input_wave_lattice_r
                      : std::max(1, static_cast<int>(std::sqrt(static_cast<float>(requested_num_probes) / max_dim))) );
      
      // Adjust the number of probes in x and y direction to ensure that it is divisible
      // by r (ensuring therefore that the lattices are compatible with the implicit
      // periodic continuation of the specimen)
      num_probes_x += r - (((num_probes_x - 1) % r) + 1);
      num_probes_y += r - (((num_probes_y - 1) % r) + 1);
      
      // Initialize the probe lattice
      probe_lattice.X = num_probes_x;
      probe_lattice.Y = num_probes_y;
      
      probe_lattice.dx = data.specimen.lenX / probe_lattice.X;
      probe_lattice.dy = data.specimen.lenY / probe_lattice.Y;
      
      probe_lattice.offset_x = 0;
      probe_lattice.offset_y = 0;
      
      // Initialize the input wave lattice
      if (data.p.inputwave == MS_InputWave::Probe) {
        // In case compatible_probe_lattice is true
        input_wave_lattice = probe_lattice;
      } else {
        input_wave_lattice.X = num_probes_x / r;
        input_wave_lattice.Y = num_probes_y / r;
        
        input_wave_lattice.dx = data.specimen.lenX / input_wave_lattice.X;
        input_wave_lattice.dy = data.specimen.lenY / input_wave_lattice.Y;
        
        input_wave_lattice.offset_x = probe_lattice.dx / 2;
        input_wave_lattice.offset_y = probe_lattice.dy / 2;
      }
      
      } break;
    case MS_InputWave::Pixel: {
      // The input waves are a single pixel in the simulation window, so
      // the simulation window itself is the input wave lattice
      int num_probes_x = static_cast<int>(std::ceil(data.specimen.lenX / data.p.req_probe_step_x));
      int num_probes_y = static_cast<int>(std::ceil(data.specimen.lenY / data.p.req_probe_step_y));
      
      // Adjust the number of probes in x and y direction to ensure that the
      // number of simulation window pixels is divisible by the number of
      // probes
      int s = static_cast<int>(data.simulation_window.X / num_probes_x);
      for (; s>1 && data.simulation_window.X % s != 0; s--);
      num_probes_x = data.simulation_window.X / s;
      
      s = static_cast<int>(data.simulation_window.Y / num_probes_y);
      for (; s>1 && data.simulation_window.Y % s != 0; s--);
      num_probes_y = data.simulation_window.Y / s;
      
      // Initialize the probe lattice
      probe_lattice.X = num_probes_x;
      probe_lattice.Y = num_probes_y;
      
      probe_lattice.dx = data.specimen.lenX / probe_lattice.X;
      probe_lattice.dy = data.specimen.lenY / probe_lattice.Y;
      
      probe_lattice.offset_x = 0;
      probe_lattice.offset_y = 0;
      
      // Initialize the input wave lattice
      input_wave_lattice.X = data.simulation_window.X;
      input_wave_lattice.Y = data.simulation_window.Y;
      
      input_wave_lattice.dx = data.specimen.lenX / input_wave_lattice.X;
      input_wave_lattice.dy = data.specimen.lenY / input_wave_lattice.Y;
      
      input_wave_lattice.offset_x = 0;
      input_wave_lattice.offset_y = 0;
      } break;
    case MS_InputWave::FourierSpaceDirac: {
			// This is a special case in that a position in Fourier space is
			// assigned to each input wave, but no position in real space.
			
			// For this type of input wave the specimen width and height does
			// not necessarily need to be an integer multiple of the probe
			// step size. However, the probe lattice is computed in the same
			// way as above to facilitate comparisons of the results.
			probe_lattice.X = static_cast<int>(std::ceil(data.specimen.lenX / data.p.req_probe_step_x));
			probe_lattice.Y = static_cast<int>(std::ceil(data.specimen.lenY / data.p.req_probe_step_y));
			
			probe_lattice.dx = data.specimen.lenX / probe_lattice.X;
			probe_lattice.dy = data.specimen.lenY / probe_lattice.Y;
			
			probe_lattice.offset_x = 0;
			probe_lattice.offset_y = 0;
			
			// The input wave lattice for this input wave is located in Fourier
			// space and not (!) in real space as for the other input waves.
			// Its dimensions are given by the maximum probe frequency
			// and the Fourier space pixel size.
			const RealType max_freq = data.p.alpha_max / data.p.lambda;
			const RealType fs_pixel_size_x = 1 / (data.simulation_window.pixel_size_x * data.simulation_window.X);
			const RealType fs_pixel_size_y = 1 / (data.simulation_window.pixel_size_y * data.simulation_window.Y);
			
			if (data.p.input_wave_fsdirac_use_ideal_frequencies && data.p.input_wave_fsdirac_px_beyond_aperture != 0)
			  Error("The \"input_wave_fsdirac_px_beyond_aperture\" parameter must be equal"
			        " to zero if \"input_wave_fsdirac_use_ideal_frequencies\" is true!");
			
			input_wave_lattice.X = 2 * std::ceil(max_freq / fs_pixel_size_x) + 2 * data.p.input_wave_fsdirac_px_beyond_aperture;
			input_wave_lattice.Y = 2 * std::ceil(max_freq / fs_pixel_size_y) + 2 * data.p.input_wave_fsdirac_px_beyond_aperture;
			
			input_wave_lattice.X = std::min(input_wave_lattice.X, data.simulation_window.X);
			input_wave_lattice.Y = std::min(input_wave_lattice.Y, data.simulation_window.Y);
			
			input_wave_lattice.dx = fs_pixel_size_x;
			input_wave_lattice.dy = fs_pixel_size_y;
			
			input_wave_lattice.offset_x = 0;
			input_wave_lattice.offset_y = 0;
		  } break;
    default:
      Error("Not implemented!", __FILE__, __LINE__);
  }
}

// Calculates the coefficient sets and indices for the approximation of the probes
// by linear combinations of the input waves
//
// The arrays approximation_errors_euc_p and approximation_errors_sup_p,
// if provided, will be filled with an image of size r x r with the
// approximation errors in the euclidean resp. supremum norm, where r x r
// is the size of the CoefficientSets and CoefficientIndices arrays.
void initProbeApproximation(const MSData& data,
                            const Lattice& probe_lattice,
                            const Lattice& input_wave_lattice,
                            Image<Image<complex>>& CoefficientSets,
                            Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                            const int max_num_input_waves,
                            const bool save_output,
                            Image<RealType> *approximation_errors_euc_p = nullptr,
                            Image<RealType> *approximation_errors_sup_p = nullptr,
                            const bool full_image_approximation_errors = true) {
  Image<RealType> approximation_errors_euc;
  Image<RealType> approximation_errors_sup;
  
  switch (data.p.inputwave) {
    case MS_InputWave::Probe: {
      // Standard Multislice algorithm: directly transmit the probe at each probe position
      // through the specimen -- no approximation necessary
      std::cerr << "Setting probe approximation coefficients for MS_InputWave::Probe ..." << std::endl;
      
      Image<complex> coeff(1, 1, {1, 0});
      std::vector<std::array<int, 2>> indices(1, {0, 0});
      
      CoefficientSets = Image(1, 1, coeff);
      CoefficientIndices = Image(1, 1, indices);
      
      approximation_errors_euc = Image<RealType>(1, 1, 0);
      approximation_errors_sup = Image<RealType>(1, 1, 0);
      } break;
    case MS_InputWave::TrigonometricPolynomial:
    case MS_InputWave::TrigonometricPolynomialV2:
    case MS_InputWave::Square:
    case MS_InputWave::Disk:
    case MS_InputWave::Gaussian:
    case MS_InputWave::ProbeSubset: {
      // Modification of the standard Multislice algorithm: use a smaller number of probe
      // positions based on the number of pixels within the objective aperture in Fourier
      // space
      if (data.p.inputwave == MS_InputWave::ProbeSubset)
	    	std::cerr << "Calculating probe approximations for MS_InputWave::ProbeSubset ..." << std::endl;
      else if (data.p.inputwave == MS_InputWave::TrigonometricPolynomial)
		    std::cerr << "Calculating probe approximations for MS_InputWave::TrigonometricPolynomial ..." << std::endl;
      else if (data.p.inputwave == MS_InputWave::TrigonometricPolynomialV2)
	    	std::cerr << "Calculating probe approximations for MS_InputWave::TrigonometricPolynomialV2 ..." << std::endl;
      else if (data.p.inputwave == MS_InputWave::Square)
	    	std::cerr << "Calculating probe approximations for MS_InputWave::Square ..." << std::endl;
      else if (data.p.inputwave == MS_InputWave::Disk)
	    	std::cerr << "Calculating probe approximations for MS_InputWave::Disk ..." << std::endl;
      else if (data.p.inputwave == MS_InputWave::Gaussian)
		    std::cerr << "Calculating probe approximations for MS_InputWave::Gaussian ..." << std::endl;
      
      if (data.propagation_window.X > data.simulation_window.X ||
          data.propagation_window.Y > data.simulation_window.Y)
        std::cerr << "############################################################" << std::endl
                  << std::endl
                  << "WARNING: the propagation window size exceeds the size of the" << std::endl
                  << "         simulation window in at least one direction. This  " << std::endl
                  << "         may cause wrong results or noticeable artifacts in " << std::endl
                  << "         the final STEM image. Consider increasing the size " << std::endl
                  << "         of the simulation window by increasing the tile_x  " << std::endl
                  << "         or tile_y parameters. Another possibility is to    " << std::endl
                  << "         decrease the input wave radius (if possible and    " << std::endl
                  << "         reasonable) or to decrease the probe_radius        " << std::endl
                  << "         parameter.                                         " << std::endl
                  << std::endl
                  << "############################################################" << std::endl
                  << std::endl;
      
      // Factor by which the number of probes that are transmitted through the specimen is
      // reduced in x and y direction
      const int r = probe_lattice.X / input_wave_lattice.X;
      
      // Calculate the coefficients for all r^2 different configurations and the indices
      // of those coefficients that will be used for the approximation of the probes as
      // linear combinations of the input waves. The number of indices is chosen as small
      // as possible based on data.p.max_probe_approximation_error
      Image<complex> coeff(input_wave_lattice.X, input_wave_lattice.Y, {0, 0});
      std::vector<std::array<int, 2>> indices;
      
      CoefficientSets = Image(r, r, coeff);
      CoefficientIndices = Image(r, r, indices);
      
      const RealType thread_MB = estimateProbeApproximationMB(data, input_wave_lattice, max_num_input_waves);
      const int max_num_threads = std::min(std::max(1, static_cast<int>(data.p.memory_limit_MB / thread_MB)), std::min(r*r, data.p.max_num_threads));
      
      std::cerr << "\t                              Total number of configurations: " << r << "^2" << std::endl
                << "\tApprox. required MB for the computation of one configuration: " << thread_MB << std::endl
                << "\t                                         Downsampling factor: " << data.p.approximation_downsampling_factor << std::endl
                << "\t     Maximum number of threads to compute the approximations: " << max_num_threads << std::endl
                << "\t                                  Target approximation error: " << 100 * data.p.max_probe_approximation_error << "% (sup norm)" << std::endl
                << std::endl;
      
      std::cerr << "\tProgress: " << 0 << " / " << r * r << std::endl;
      
      approximation_errors_euc = Image<RealType>(r, r, -1);
      approximation_errors_sup = Image<RealType>(r, r, -1);
      
      int progress = 0;
      #pragma omp parallel for collapse(2) num_threads(max_num_threads)
      for (int j=0; j<r; j++)
        for (int i=0; i<r; i++) {
          // Output for the probe approximation is generated only for one case
          const bool generate_output = (j==0 && i==0 && save_output);
          
          ProbeApproximation(data,
                             {i, j},
                             probe_lattice,
                             input_wave_lattice,
                             CoefficientSets(i, j),
                             CoefficientIndices(i, j),
                             generate_output,
                             max_num_input_waves,
                             approximation_errors_euc(i, j),
                             approximation_errors_sup(i, j));
          
          #pragma omp critical (initProbeApproximation)
          {
            std::cerr << "\t          " << ++progress << " / " << r * r << " : " << i << ", " << j << std::endl
                      << "\t\t          " << "Downsampling factor: " << data.p.approximation_downsampling_factor << std::endl
                      << "\t\t          " << "Approximation error: " << approximation_errors_sup(i, j) << " (sup norm, downsampled images)" << std::endl
                      << "\t\t          " << "Number of input waves: " << CoefficientIndices(i, j).size() << std::endl;
          }
        }
      std::cerr << std::endl;
      
      } break;
    case MS_InputWave::Pixel: {
      // There is one input wave for every pixel of the simulation window, each
      // corresponding to the indicator function of that pixel
      std::cerr << "Setting probe approximation coefficients for MS_InputWave::Pixel ..." << std::endl;
      
      Image<complex> coeff(input_wave_lattice.X, input_wave_lattice.Y, {0, 0});
      std::vector<std::array<int, 2>> indices;
      
      CoefficientSets = Image(1, 1, coeff);
      CoefficientIndices = Image(1, 1, indices);
      
      // Calculate an image of the probe (the pixel values of which are the coefficients)
      AlignedArray probe_window = getProbeWindow(data, {0, 0});
      
      Image<complex> probe = data.input.probe.get2DImageNonSquare(data.simulation_window.pixel_size_x,
                                                                  data.simulation_window.pixel_size_y,
                                                                  probe_window.aa_size[0],
                                                                  probe_window.aa_size[1],
                                                                  probe_window.rel_wave_position[0],
                                                                  probe_window.rel_wave_position[1],
                                                                  {0, 0});
      
      if (probe.getX() > input_wave_lattice.X || probe.getY() > input_wave_lattice.Y)
        Error("Cannot use input wave type \"Pixel\" if the probe window size ("
              + std::to_string(probe.getX()) + " x " + std::to_string(probe.getY())
              + ") exceeds the simulation window size (" + std::to_string(input_wave_lattice.X)
              + " x " + std::to_string(input_wave_lattice.Y) + ")!");
      
      // Add values to the coefficient sets and indices
      CoefficientIndices(0, 0).reserve(probe.getX() * probe.getY());
      for (int y=0; y<probe.getY(); y++)
		    for (int x=0; x<probe.getX(); x++) {
		      const int lattice_x = normalize_periodic(x - probe.getX()/2, input_wave_lattice.X);
		      const int lattice_y = normalize_periodic(y - probe.getY()/2, input_wave_lattice.Y);
		      
		      CoefficientSets(0, 0)(lattice_x, lattice_y) = probe(x, y);
		      CoefficientIndices(0, 0).push_back({lattice_x, lattice_y});
		    }
      
      approximation_errors_euc = Image<RealType>(1, 1, 0);
      approximation_errors_sup = Image<RealType>(1, 1, 0);
      } break;
    case MS_InputWave::FourierSpaceDirac: {
      // The approximation coefficients are different for every probe position
      // and will be computed as modulations of the Fourier space probe
      // pixel values for each probe position
      std::cerr << "Setting probe approximation coefficients for MS_InputWave::FourierSpaceDirac ..." << std::endl;
      
      if (data.p.input_wave_fsdirac_use_ideal_frequencies)
        std::cerr << "\tUsing ideal Fourier space frequencies." << std::endl << std::endl;
      else
        std::cerr << "\tUsing the Fourier space frequencies of a simulated probe image." << std::endl << std::endl;
      
      // Calculate a sample image of the probe for the computation of the
      // scaling factor for the approximation below
      std::array<RealType, 2> probe_pos = probe_lattice.getPosition(0, 0);
      AlignedArray probe_window = getProbeWindow(data, probe_pos);
      
			Image<complex> probe = data.input.probe.get2DImageNonSquare(
												 data.simulation_window.pixel_size_x,
												 data.simulation_window.pixel_size_y,
												 probe_window.aa_size[0],
												 probe_window.aa_size[1],
												 probe_window.rel_wave_position[0],
												 probe_window.rel_wave_position[1],
												 {0, 0}
											 );
			
			// Calculate an image of the Fourier space probe frequencies
			// (this is used to set the approximation coefficients below if
			//  data.p.input_wave_fsdirac_use_ideal_frequencies == false)
			Image<complex> probe_fs(data.simulation_window.X, data.simulation_window.Y, {0, 0});
			if (!data.p.input_wave_fsdirac_use_ideal_frequencies) {
				Image<complex> probe_rs = data.input.probe.get2DImageNonSquare(
													 data.simulation_window.pixel_size_x,
													 data.simulation_window.pixel_size_y,
													 data.simulation_window.X,
													 data.simulation_window.Y,
													 {0, 0});
				
				probe_rs.applyFourierShift();
				FourierTransform(probe_rs, &probe_fs);
				probe_fs.applyFourierShift();
			}
      
      // Set CoefficientSets to the Fourier space probe values within the aperture
      // and CoefficientIndices to a vector of the pixels of all discrete frequencies
      // considered for the approximation (i.e. every f-th frequency in each
      // direction within the aperture radius, where f = data.p.input_wave_lattice_r).
      //
      // Note that the values in CoefficientSets need to be modulated appropriately
      // for probe positions different from {0, 0}
      const RealType max_freq = data.p.alpha_max / data.p.lambda;
			const RealType fs_pixel_size_x = 1 / (data.simulation_window.pixel_size_x * data.simulation_window.X);
			const RealType fs_pixel_size_y = 1 / (data.simulation_window.pixel_size_y * data.simulation_window.Y);
			
			const int max_x = input_wave_lattice.X / 2;
			const int max_y = input_wave_lattice.Y / 2;
			
      Image<complex> coeff(input_wave_lattice.X, input_wave_lattice.Y, {0, 0});
      std::vector<std::array<int, 2>> indices;
      
      Image<complex> linear_combination(probe_window.aa_size[0], probe_window.aa_size[1], {0, 0});
			for (int y=-max_y; y<max_y; y++)
			  for (int x=-max_x; x<max_x; x++)
					if (x % data.p.input_wave_lattice_r == 0 &&
					    y % data.p.input_wave_lattice_r == 0) {
					  const RealType dx = x * fs_pixel_size_x;
					  const RealType dy = y * fs_pixel_size_y;
					  const RealType dsqr = dx*dx + dy*dy;
					  
					  const RealType ext_dist_x = data.p.input_wave_fsdirac_px_beyond_aperture * fs_pixel_size_x;
					  const RealType ext_dist_y = data.p.input_wave_fsdirac_px_beyond_aperture * fs_pixel_size_y;
					  const RealType ext_dist_sqr = ext_dist_x * ext_dist_x + ext_dist_y * ext_dist_y;
					  
					  if (dsqr - ext_dist_sqr < max_freq*max_freq) {
							if (data.p.input_wave_fsdirac_use_ideal_frequencies) {
							  const RealType chi = aberrationFunction(data.p, dsqr);
							  coeff(x+max_x, y+max_y) = complex(cos(chi), -sin(chi));
							} else {
							  coeff(x+max_x, y+max_y) = probe_fs(probe_fs.getX()/2+x, probe_fs.getY()/2+y);
							}
							
					    indices.push_back({x, y});
					    
					    AlignedArray input_window = getInputWaveWindow(data, {0, 0} /* irrelevant */, {x, y}, {0, 0} /* irrelevant */);
					    Image<complex> input_wave = data.input.getInitialCondition(input_window);
					    input_wave *= coeff(x+max_x, y+max_y);
					    linear_combination += input_wave.getPeriodic(probe_window.aa_size[0],
						                                               probe_window.aa_size[1],
						                                               probe_window.aa_pos[0],
						                                               probe_window.aa_pos[1]);
					  }
			    }
			
			// Calculate a real-valued scaling factor for the best L2-Approximation of
			// the probe image by the linear combination of input waves computed above
      const RealType alpha = sum(probe).real() / sum(linear_combination).real();
      
      coeff *= alpha;
			
      CoefficientSets = Image(1, 1, coeff);
			CoefficientIndices = Image(1, 1, indices);
			
			// [The approximation errors are computed below and not here]
      approximation_errors_euc = Image<RealType>(1, 1, -1);
      approximation_errors_sup = Image<RealType>(1, 1, -1);
      
      } break;
    default:
      Error("Not implemented!", __FILE__, __LINE__);
  }
  
  // Calculate the approximation errors for the non-downsampled images
  // and save the resulting probe approximations to file
  //
  // For an explanation of the steps below, compare with the CalculateLinearCombination()
  // method of LatticeMultisliceAlgorithm
  const bool gen_output = (data.p.writeProbeApproximationOutput && save_output);
  const bool calc_errors = ((full_image_approximation_errors && data.p.approximation_downsampling_factor != 1) || data.p.inputwave == MS_InputWave::FourierSpaceDirac)
                           && (approximation_errors_euc_p != nullptr || approximation_errors_sup_p != nullptr);
  if (gen_output || calc_errors) {
    if (gen_output)
      std::cerr << "Generating and saving the resulting probe approximations ..." << std::endl;
    
    std::cerr << "\tApproximation errors on the full resolution probe window (sup norm):" << std::endl;
    #pragma omp parallel for collapse(2) num_threads(data.p.max_num_threads)
    for (int j=0; j<CoefficientSets.getY(); j++)
      for (int i=0; i<CoefficientSets.getX(); i++) {
        // Get the Probe position in Angstrom
        std::array<RealType, 2> probe_pos = probe_lattice.getPosition(i, j);
        
        AlignedArray probe_window = getProbeWindow(data, probe_pos);
        
        // Calculate the linear combination of the input waves approximating
        // the probe
        Image<complex> result(probe_window.aa_size[0], probe_window.aa_size[1], {0, 0});
        for (std::array<int, 2> coeff_coord: CoefficientIndices(i, j)) {
          // Get the input wave coordinates in the input wave lattice
          // (Remark: here, iw_coord will always be equal to coeff_coord
          //          unless the FourierSpaceDirac input wave is used)
          const std::array<int, 2> iw_coord = getInputWaveCoord({i, j}, coeff_coord, input_wave_lattice, probe_lattice, data.p);
          
          // Get the input wave position in Angstrom
          std::array<RealType, 2> input_wave_pos = input_wave_lattice.getPosition(iw_coord[0], iw_coord[1]);
          
          // Get the input wave window coordinates
          AlignedArray input_window = getInputWaveWindow(data, input_wave_pos, coeff_coord, probe_pos);
          
          // Calculate the current summand ...
          Image<complex> input_wave = data.input.getInitialCondition(input_window);
          
          input_wave *= CoefficientSets(i, j)(iw_coord[0], iw_coord[1]);
          
          // ... and add it to the result
          if (data.p.inputwave == MS_InputWave::FourierSpaceDirac) {
						result += input_wave.getPeriodic(probe_window.aa_size[0],
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
								 += input_wave(input_window_start[0] + x, input_window_start[1] + y);
					}
        }
        
        // Calculate an image of the probe
        Image<complex> probe = data.input.probe.get2DImageNonSquare(
                           data.simulation_window.pixel_size_x,
                           data.simulation_window.pixel_size_y,
                           probe_window.aa_size[0],
                           probe_window.aa_size[1],
                           probe_window.rel_wave_position[0],
                           probe_window.rel_wave_position[1],
                           {0, 0}
                         );
        
        // Update the relative approximation error arrays
        Image<complex> diff(result);
        diff -= probe;
        
        approximation_errors_euc(i, j) = norm(diff) / norm(probe);
        approximation_errors_sup(i, j) = sup_norm(diff) / sup_norm(probe);
        
        #pragma omp critical (initProbeApproximation2)
        {
          if (gen_output) {
            // Save the resulting linear combination of the input waves (which approximates the probe)
            save(result,
                 std::to_string(i) + "_" + std::to_string(j) + "_approximation",
                 data.p.outputDir + "/ProbeApproximation/Result");
            
            // Save an image of the probe
            save(probe,
                 std::to_string(i) + "_" + std::to_string(j) + "_probe",
                 data.p.outputDir + "/ProbeApproximation/Result");
            
            // Save the pointwise absolute value of the difference abs_diff
            Image<RealType> abs_diff(diff.getX(), diff.getY(), 0);
            for (int k=0; k<diff.size(); k++)
              abs_diff[k] = sqrt(abs_sqr(diff[k]));
            
            abs_diff /= sup_norm(probe);
            
            save(abs_diff,
                 std::to_string(i) + "_" + std::to_string(j) + "_abs_diff_normalized",
                 data.p.outputDir + "/ProbeApproximation/Result");
            
            // Save the pointwise absolute value of the difference of the Fourier transforms
            Image<complex> diff_fs(diff);
            FourierTransform(diff, &diff_fs);
            diff_fs.applyFourierShift();
            
            Image<RealType> abs_diff_fs(diff.getX(), diff.getY(), 0);
            for (int k=0; k<diff.size(); k++)
              abs_diff_fs[k] = sqrt(abs_sqr(diff_fs[k]));
            
            save(abs_diff_fs,
                 std::to_string(i) + "_" + std::to_string(j) + "_abs_diff_fs",
                 data.p.outputDir + "/ProbeApproximation/Result");
          }
          
          std::cerr << "\t\t(" << i << ", " << j << "): " << approximation_errors_sup(i, j) << std::endl;
        }
      }
    
    std::cerr << std::endl;
  }
  
  // Print a warning if there is a large spread in the sup-norm approximation errors
  // as this may cause noticeable artifacts in the final STEM image
  Image<RealType> error_spread_matrix(approximation_errors_sup.getX(),
                                      approximation_errors_sup.getY(),
                                      sup_norm(approximation_errors_sup));
  
  error_spread_matrix -= approximation_errors_sup;
  
  const RealType max_error_diff = sup_norm(error_spread_matrix);
  if (max_error_diff > 0.005) {
    std::cerr << "WARNING: there is a large difference in the probe approximation" << std::endl
              << "         errors between the different configurations (maximum  " << std::endl
              << "         difference " << max_error_diff << ", supremum norm). This may cause" << std::endl
              << "         noticeable, grid-like artifacts in the output STEM    " << std::endl
              << "         image due to the regular spacing of different-grade   " << std::endl
              << "         approximations. This problem can usually be avoided by" << std::endl
              << "         slightly changing the number of input waves used in   " << std::endl
              << "         the approximations.                                   " << std::endl
              << std::endl;
  }
  
  // Always print a note listing several options to improve the probe approximation
  std::cerr << "---------------------------------------------------------------" << std::endl
            << "Note: if the approximation results are unsatisfactory, consider" << std::endl
            << "      trying one of the following approaches:                  " << std::endl
            << " (1) decrease the max_probe_approximation_error parameter      " << std::endl
            << " (2) increase the propagation_window_size parameter            " << std::endl
            << " (3) increase the approximation_max_input_waves parameter      " << std::endl
            << " (4) decrease the input_wave_lattice_r parameter               " << std::endl
            << " (5) adjust input wave specific parameters such as for example " << std::endl
            << "     trig_poly_degree or input_wave_gaussian_sigma             " << std::endl
            << " (6) inspect results by setting writeProbeApproximationOutput  " << std::endl
            << "     to true                                                   " << std::endl
            << "---------------------------------------------------------------" << std::endl
            << std::endl;
  
  if (approximation_errors_euc_p != nullptr)
    *approximation_errors_euc_p = approximation_errors_euc;
  
  if (approximation_errors_sup_p != nullptr)
    *approximation_errors_sup_p = approximation_errors_sup;
}

// Calculates the STEM detector values from a given exit wave, which is expected to be in
// real space coordinates (and not in Fourier space) with the dimensions lenX x lenY in
// Angstrom
std::vector<RealType> calculateDetectorValues(const Image<complex>& exit_wave,
                                              const RealType lenX,
                                              const RealType lenY,
                                              const Param& p,
                                              const int numDetectors) {
  const int X = exit_wave.getX();
  const int Y = exit_wave.getY();
  
  std::vector<RealType> detector_values(numDetectors, static_cast<RealType>(0));
  
  // Convert the exit wave to Fourier space
  Image<complex> ew_rs(exit_wave);
  Image<complex> ew_fs(exit_wave);
  FourierTransform(ew_rs, &ew_fs);
  ew_fs.applyFourierShift();
  
  // Integrate the Fourier space exit wave in intervals of scattering angles
  // (= radii in the Fourier space image ew_fs)
  for (int y=0; y<Y; y++)
    for (int x=0; x<X; x++) {
      // Calculate the scattering angle corresponding to (x, y) in mrad
      const RealType dx = x - X/2;
      const RealType dy = y - Y/2;
      
      const RealType scattering_angle = p.lambda * sqrt(dx*dx / (lenX*lenX) + dy*dy / (lenY*lenY)) * 1000;
      
      // Assign to a detector based on the scattering angle
      const int bin = static_cast<int>(scattering_angle / p.detector_stepsize);
      if (bin < numDetectors)
        detector_values[bin] += ew_fs(x, y).real() * ew_fs(x, y).real() + ew_fs(x, y).imag() * ew_fs(x, y).imag();
    }
  
  // Divide by sqrt(X*Y) to account for the forward Fourier transform above. This
  // normalization factor is squared because the detector measures the absolute value
  // squared of the electron wave (equivalently, ew_fs above could have been divided by
  // sqrt(X*Y))
  for (RealType& val: detector_values)
    val /= X * Y;
  
  return detector_values;
}

// Calculate all probe positions within the Output window that are
// close enough to one of the locations where the specimen changed
// such that a recomputation of this probe position is necessary
std::vector<std::array<int, 2>> getProbeUpdatePositions(const MSData& data,
                                                        const std::vector<std::array<RealType, 3>>& change_locations,
                                                        const Lattice& probe_lattice,
                                                        const OutputWindow& output_window) {
  // Calculate an upper bound for the distance in x and y direction from
  // a change in the specimen that requires a recomputation of a point
  // in the probe lattice
  const RealType probe_window_lenX = data.probe_window_size[0] * data.propagation_window.pixel_size_x;
  const RealType probe_window_lenY = data.probe_window_size[1] * data.propagation_window.pixel_size_y;
  
  const RealType max_change_location_dist_x = data.p.potential_bound + probe_window_lenX / 2;
  const RealType max_change_location_dist_y = data.p.potential_bound + probe_window_lenY / 2;
  
  // Iterate over all pixels of the output window and add all those with
  // a small distance to any element of change_locations to the returned
  // vector
  std::vector<std::array<int, 2>> res;
  res.reserve(output_window.X * output_window.Y);
  
  for (int y=0; y<output_window.Y; y++)
    for (int x=0; x<output_window.X; x++) {
      std::array<RealType, 2> probe_pos = probe_lattice.getPosition(x + output_window.start_x, y + output_window.start_y);
      
      for (const std::array<RealType, 3>& loc: change_locations) {
        const RealType dist_x = std::abs(loc[0] - probe_pos[0]);
        const RealType dist_y = std::abs(loc[1] - probe_pos[1]);
        
        // Note: we add 3*loc[2] = 3 * <standard deviation of random thermal motion of the atoms>
        //       to account for the atom displacement in the frozen phonon approximation
        //
        // If data.p.frozen_phonon_iterations == 1, then loc[2] will be zero as set in
        // updateTransmissionFunctions() in CPU/MSData.h
        if (dist_x < max_change_location_dist_x + 3*loc[2] && dist_y < max_change_location_dist_y + 3*loc[2]) {
          res.push_back({x, y});
          break;
        }
      }
    }
  
  return res;
}

// Calculate all input wave positions for which a Multislice result
// needs to be computed or recomputed. This may either be because a
// Multislice result is not available, or because a Multislice result is
// available, but needs to be recomputed because of changes in the
// specimen.
//
// Also calculates a vector of indices of entries of MultisliceResult,
// which are no longer needed, to_delete.
std::vector<std::array<int, 2>> getInputWaveUpdatePositions(const MSData& data,
                                                            const std::vector<std::array<int, 2>>& probe_pos,
                                                            const std::vector<std::array<RealType, 3>>& change_locations,
                                                            const Lattice& input_wave_lattice,
                                                            const Lattice& probe_lattice,
                                                            const OutputWindow& output_window,
                                                            const Image<Image<complex>>& MultisliceResult,
                                                            const Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                                                            std::vector<std::array<int, 2>>& to_delete) {
  // Calculate an upper bound for the distance in x and y direction from
  // a change in the specimen that requires a recomputation of an entry
  // of MultisliceResult
  const AlignedArray input_window = getInputWaveWindow(data, {0, 0}, {0, 0}, {0, 0});
  
  const RealType input_wave_window_lenX = input_window.aa_size[0] * data.propagation_window.pixel_size_x;
  const RealType input_wave_window_lenY = input_window.aa_size[1] * data.propagation_window.pixel_size_y;
  
  const RealType max_change_location_dist_x = data.p.potential_bound + input_wave_window_lenX / 2;
  const RealType max_change_location_dist_y = data.p.potential_bound + input_wave_window_lenY / 2;
  
  // Get all input waves required for the computation of all probe
  // positions in probe_pos
  const std::vector<std::array<int, 2>> all_input_waves = getRequiredInputWavePositions(probe_pos,
                                                                                        CoefficientIndices,
                                                                                        input_wave_lattice,
                                                                                        probe_lattice,
                                                                                        output_window.start_x,
                                                                                        output_window.start_y,
                                                                                        data.p);
  
  const int exit_wave_pixel = input_window.aa_size[0] * input_window.aa_size[1];
  const RealType ew_MB = exit_wave_pixel * sizeof(complex) / static_cast<RealType>(1024 * 1024);
  
  if (static_cast<int>(data.p.memory_limit_MB / ew_MB) < static_cast<int>(all_input_waves.size())) {
    std::cerr << "WARNING: the recomputation of local changes requires at least " << static_cast<int>(all_input_waves.size() * ew_MB) << " MB" << std::endl
              << "         of computer memory. This exceeds the requested memory limit" << std::endl
              << "         of " << data.p.memory_limit_MB << " MB." << std::endl
              << std::endl
              << "Note: only the trivial LMA strategy is available for the recomputation" << std::endl
              << "      of local changes." << std::endl
              << std::endl;
  }
  
  // Iterate through all input waves and determine if the corresponding
  // entry of MultisliceResult needs to be (re)computed
  std::vector<std::array<int, 2>> to_compute;
  to_compute.reserve(all_input_waves.size());
  
  for (const std::array<int, 2>& iw_coord: all_input_waves) {
    // If the corresponding Multislice result is not available, it needs
    // to be computed in any case, independent of the distance from
    // changes in the specimen
    if (MultisliceResult(iw_coord[0], iw_coord[1]).size() == 0) {
      to_compute.push_back(iw_coord);
      continue;
    }
    
    // For the FourierSpaceDirac input wave, all Multislice results need
    // to be recomputed
    if (data.p.inputwave == MS_InputWave::FourierSpaceDirac) {
			to_compute.push_back(iw_coord);
			continue;
		}
    
    // Otherwise, check if the distance to any change in the specimen is
    // smaller than max_change_location_dist_{x|y} in both directions
    const std::array<RealType, 2> iw_pos = input_wave_lattice.getPosition(iw_coord[0], iw_coord[1]);
    
    for (const std::array<RealType, 3>& loc: change_locations) {
      const RealType dist_x = std::abs(loc[0] - iw_pos[0]);
      const RealType dist_y = std::abs(loc[1] - iw_pos[1]);
      
      // Note: we add 3*loc[2] = 3 * <standard deviation of random thermal motion of the atoms>
      //       to account for the atom displacement in the frozen phonon approximation
      //
      // If data.p.frozen_phonon_iterations == 1, then loc[2] will be zero as set in
      // updateTransmissionFunctions() in CPU/MSData.h
      if (dist_x < max_change_location_dist_x + 3*loc[2] && dist_y < max_change_location_dist_y + 3*loc[2]) {
        to_compute.push_back(iw_coord);
        break;
      }
    }
  }
  
  // Find all previously computed entries of MultisliceResult that are not in
  // all_input_waves
  to_delete.clear();
  
  std::vector<std::array<int, 2>> prev_iw_pos;
  for (int y=0; y<input_wave_lattice.Y; y++)
	for (int x=0; x<input_wave_lattice.X; x++)
	  if (MultisliceResult(x, y).size() != 0)
	    prev_iw_pos.push_back({x, y});
  
  auto cmp = [&](const std::array<int, 2>& a, const std::array<int, 2>& b) {
    return (a[0] + a[1] * input_wave_lattice.X < b[0] + b[1] * input_wave_lattice.X);
  };
  
  std::sort(prev_iw_pos.begin(), prev_iw_pos.end(), cmp);
  
  std::set_difference(prev_iw_pos.cbegin(), prev_iw_pos.cend(),
                      all_input_waves.cbegin(), all_input_waves.cend(),
                      std::inserter(to_delete, to_delete.begin()),
                      cmp);
  
  return to_compute;
}

#endif  // multislice_latticemultislicealgorithm_aux_h
