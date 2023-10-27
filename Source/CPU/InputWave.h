#ifndef multislice_cpu_inputwave_h
#define multislice_cpu_inputwave_h

#include "InputWave_aux.h"

#include "SimulationWindow.h"
#include "PropagationWindow.h"

#include "../Utility/Image.h"
#include "../Utility/CircularImage.h"
#include "../Utility/AlignedArray.h"
#include "../Utility/Error.h"

// Stores the probe wave function and the initial condition to the Schrödinger equation
// that is used for the Multislice computations
struct InputWave {
  // The probe wave function as a high-resolution circular image
  CircularImage<complex> probe;
  
  // The initial condition (= input wave) to the Schrödinger equation.
  // Either given as a circular image or as a standard 2D array
  struct InitialCondition {
    CircularImage<complex> circular_image;
    Image<complex> standard_image;
  } initial_condition;
  
  // Same as probe radius, but for the initial condition; in Angstrom
  RealType initial_condition_radius;
  
  // Pixel size used for the computations in init() below, in Angstrom
  RealType hires_pixel_size;
  
  // The input wave type
  MS_InputWave inputwave_type;
  
  void init(const Param& p, const SimulationWindow& simulation_window) {
		inputwave_type = p.inputwave;
    
    // High resolution pixel size for the supersampled images
    hires_pixel_size = std::min(simulation_window.pixel_size_x, simulation_window.pixel_size_y) / p.hires_supersampling_factor;
    
    // Calculate the STEM probe wave function
    probe = getPropagatedProbe(p, hires_pixel_size, 0);
    
    // Set the initial condition
    switch (p.inputwave) {
      case MS_InputWave::Probe:
      case MS_InputWave::ProbeSubset:
        initial_condition.circular_image = probe;
        initial_condition_radius = p.probe_radius;
        break;
      case MS_InputWave::TrigonometricPolynomial:
      	initial_condition.standard_image = getTrigPolyFunction(p,
      	                                                       hires_pixel_size,
      	                                                       initial_condition_radius);
        break;
      case MS_InputWave::TrigonometricPolynomialV2:
        initial_condition.circular_image = getTrigPolyFunctionCroppedFS(p,
                                                                        hires_pixel_size,
                                                                        initial_condition_radius);
        break;
      case MS_InputWave::Square:
        initial_condition.standard_image = getSquareInputWave(p, hires_pixel_size);
        initial_condition_radius = 1.5 * p.input_wave_square_len;
        break;
      case MS_InputWave::Disk:
        initial_condition.circular_image = getDiskInputWave(p, hires_pixel_size);
        initial_condition_radius = 2 * p.input_wave_disk_radius;
        break;
      case MS_InputWave::Pixel:
        initial_condition.standard_image = Image<complex>(1, 1, {1, 0});
        initial_condition_radius = p.input_wave_pixel_propagation_window_size;
        break;
      case MS_InputWave::Gaussian:
        initial_condition.circular_image = getGaussianInputWave(p,
                                                                hires_pixel_size,
                                                                initial_condition_radius);
        break;
      case MS_InputWave::FourierSpaceDirac:
        // There is nothing to do here - the input waves will be computed on the fly
        // (see the getInitialCondition functions below)
        break;
      default:
        Error("Not implemented!", __FILE__, __LINE__);
    }
  }
  
  // Returns a 2D Image of the input wave. The size of the image is
  // (X times Y) pixel, where each pixel has a size of
  // (sampling_pixel_size_x times sampling_pixel_size_y) Angstrom. The
  // center of the input wave is located at (pos_x, pos_y), given as
  // the distance from the top left corner of the returned image in
  // Angstrom.
  Image<complex> getInitialCondition(const RealType sampling_pixel_size_x,
                                     const RealType sampling_pixel_size_y,
                                     const int X,
                                     const int Y,
                                     const RealType pos_x,
                                     const RealType pos_y) const {
    // Special case MS_InputWave::Pixel
    if (inputwave_type == MS_InputWave::Pixel)
      return getPixelInputWave(sampling_pixel_size_x,
                               sampling_pixel_size_y,
                               X,
                               Y,
                               pos_x + sampling_pixel_size_x/2,
                               pos_y + sampling_pixel_size_y/2);
    
    // Special case MS_InputWave::FourierSpaceDirac
    if (inputwave_type == MS_InputWave::FourierSpaceDirac)
      return getFSDiracInputWave(X,
                                 Y,
                                 static_cast<int>(pos_x),
                                 static_cast<int>(pos_y));
    
    // Return the circular image converted to a standard 2D image ...
    if (!initial_condition.circular_image.empty())
	    return initial_condition.circular_image.get2DImageNonSquare(sampling_pixel_size_x,
	                                                                sampling_pixel_size_y,
	                                                                X,
	                                                                Y,
	                                                                pos_x,
	                                                                pos_y,
	                                                                {0, 0});
	  
	  // ... or an appropriately scaled subsection of the high-resolution 2D image
	  const RealType offset_x = (initial_condition.standard_image.getX()/2 * hires_pixel_size - pos_x) / hires_pixel_size;
	  const RealType offset_y = (initial_condition.standard_image.getY()/2 * hires_pixel_size - pos_y) / hires_pixel_size;
	  
	  return initial_condition.standard_image.resampleBilinear(sampling_pixel_size_x / hires_pixel_size,
	                                                           sampling_pixel_size_y / hires_pixel_size,
	                                                           X,
	                                                           Y,
	                                                           offset_x,
	                                                           offset_y,
	                                                           {0, 0});
  }
  
  // Same as getInitialCondition() above, but here the distance to (pos_x, pos_y)
  // is calculated with periodic continuation. This avoids the wrap-around
  // error in the Fourier space Multislice algorithm.
  Image<complex> getInitialConditionPeriodic(const RealType sampling_pixel_size_x,
                                             const RealType sampling_pixel_size_y,
                                             const int X,
                                             const int Y,
                                             const RealType pos_x,
                                             const RealType pos_y) const {
    // Special case MS_InputWave::Pixel
    if (inputwave_type == MS_InputWave::Pixel)
      return getPixelInputWave(sampling_pixel_size_x,
                               sampling_pixel_size_y,
                               X,
                               Y,
                               pos_x + sampling_pixel_size_x/2,
                               pos_y + sampling_pixel_size_y/2);
    
    // Special case MS_InputWave::FourierSpaceDirac
    if (inputwave_type == MS_InputWave::FourierSpaceDirac)
      return getFSDiracInputWave(X,
                                 Y,
                                 static_cast<int>(pos_x),
                                 static_cast<int>(pos_y));
    
    // Return the circular image converted to a standard 2D image ...
    if (!initial_condition.circular_image.empty())
	    return initial_condition.circular_image.get2DImageNonSquarePeriodic(sampling_pixel_size_x,
	                                                                        sampling_pixel_size_y,
	                                                                        X,
	                                                                        Y,
	                                                                        pos_x,
	                                                                        pos_y,
	                                                                        {0, 0});
	  
  	// ... or an appropriately scaled subsection of the high-resolution 2D image
  	const RealType offset_x = (initial_condition.standard_image.getX()/2 * hires_pixel_size - pos_x) / hires_pixel_size;
  	const RealType offset_y = (initial_condition.standard_image.getY()/2 * hires_pixel_size - pos_y) / hires_pixel_size;
	  
	  return initial_condition.standard_image.resampleBilinearPeriodic(sampling_pixel_size_x / hires_pixel_size,
	                                                                   sampling_pixel_size_y / hires_pixel_size,
	                                                                   X,
	                                                                   Y,
	                                                                   offset_x,
	                                                                   offset_y,
	                                                                   {0, 0});
  }
  
  Image<complex> getInitialCondition(const AlignedArray& aa) const {
		if (inputwave_type == MS_InputWave::FourierSpaceDirac)
		  return getFSDiracInputWave(aa.aa_size[0],
		                             aa.aa_size[1],
		                             aa.wave_index[0],
		                             aa.wave_index[1]);
		
		return getInitialCondition(aa.grid_pixel_size[0],
		                           aa.grid_pixel_size[1],
		                           aa.aa_size[0],
		                           aa.aa_size[1],
		                           aa.rel_wave_position[0],
		                           aa.rel_wave_position[1]);
	}
	
	Image<complex> getInitialConditionPeriodic(const SimulationWindow& simulation_window,
	                                           const AlignedArray& aa) const {
		if (inputwave_type == MS_InputWave::FourierSpaceDirac)
		  return getFSDiracInputWave(simulation_window.X,
		                             simulation_window.Y,
		                             aa.wave_index[0],
		                             aa.wave_index[1]);
		
		return getInitialConditionPeriodic(simulation_window.pixel_size_x,
		                                   simulation_window.pixel_size_y,
		                                   simulation_window.X,
		                                   simulation_window.Y,
		                                   aa.wave_position[0],
		                                   aa.wave_position[1]);
	}
};

#endif  // multislice_cpu_inputwave_h
