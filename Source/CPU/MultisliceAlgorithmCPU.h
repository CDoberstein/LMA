// Implementation of the Multislice algorithm on the CPU:
//
//   (1) The standard Multislice implementation that calculates the convolution with the
//       propagation function as a pointwise multiplication in Fourier space using the FFT
//
//   (2) The real space implementation that calculates the convolution with the
//       propagation function directly, without using any Fourier transforms
#ifndef multislice_cpu_multislicealgorithmcpu_h
#define multislice_cpu_multislicealgorithmcpu_h

#include "MSData.h"
#include "FFTWData.h"

#include "../Utility/Image.h"
#include "../Utility/AlignedArray.h"

#include <array>

// Implementation of the standard Fourier space Multislice algorithm
Image<complex> MultisliceAlgorithmFS(const AlignedArray& input_wave_window,
                                     FFTWData& fftw_data,
                                     const MSData& data,
                                     RealType& norm_change) {
  if (!fftw_data.validMemAddr())
	Error("Address of FFTW arrays has changed!", __FILE__, __LINE__);
  
  // Calculate the input wave window for the extended propagation window
  // in case it is needed below
  AlignedArray ext_input_wave_window(input_wave_window.wave_position,
                                     input_wave_window.wave_index,
                                     {data.propagation_window.ext_X, data.propagation_window.ext_Y},
                                     input_wave_window.grid_pixel_size);
  
  // Set the initial condition for the electron wave
  // (Important: the below if-else statement must not be simplified due
  //  to the pre-set dimensions of fftw_data.electron_wave_rs!)
  Image<complex> iw_simulation_window = data.input.getInitialConditionPeriodic(data.simulation_window,
                                                                               input_wave_window);
  
  if (data.p.fs_buffer_zone >= 0)
    fftw_data.electron_wave_rs = iw_simulation_window.getPeriodic(ext_input_wave_window.aa_size[0],
                                                                  ext_input_wave_window.aa_size[1],
                                                                  ext_input_wave_window.aa_pos[0],
                                                                  ext_input_wave_window.aa_pos[1]);
  else
    fftw_data.electron_wave_rs = iw_simulation_window;
  
  const RealType input_wave_norm = norm(fftw_data.electron_wave_rs);
  
  for (const auto& tf: data.tf.slice_transmission_function) {
    // Transmission step (pointwise multiplication with the transmission function)
    if (data.p.fs_buffer_zone >= 0)
      fftw_data.electron_wave_rs *= tf.getPeriodic(ext_input_wave_window.aa_size[0],
																									 ext_input_wave_window.aa_size[1],
																									 ext_input_wave_window.aa_pos[0],
																									 ext_input_wave_window.aa_pos[1]);
		else
		  fftw_data.electron_wave_rs *= tf;
    
    // Fourier transform
    fftw_execute(fftw_data.forward_transform);
    fftw_data.electron_wave_fs.applyFourierShift();
    
    // Propagation step (pointwise multiplication with the propagation function)
    fftw_data.electron_wave_fs *= data.pf.propagation_function_fs;
    
    // Inverse Fourier transform and normalization
    fftw_data.electron_wave_fs.applyInverseFourierShift();
    fftw_execute(fftw_data.backward_transform);
    
    if (data.p.fs_buffer_zone >= 0)
      fftw_data.electron_wave_rs /= static_cast<RealType>(ext_input_wave_window.aa_size[0] * ext_input_wave_window.aa_size[1]);
    else
      fftw_data.electron_wave_rs /= static_cast<RealType>(data.simulation_window.X * data.simulation_window.Y);
  }
  
  // Calculate <norm of exit wave> / <norm of input wave>
  norm_change = norm(fftw_data.electron_wave_rs) / input_wave_norm;
  
  // Crop the result to the 2D array given by input_wave_window, unless
  // the input wave type is FourierSpaceDirac, in which case the unchanged
  // exit wave in real space is returned (the input wave window size is
  // equal to the simulation window size in this case anyway)
  if (data.p.inputwave == MS_InputWave::FourierSpaceDirac)
    return fftw_data.electron_wave_rs;
  
  if (data.p.fs_buffer_zone >= 0)
    return fftw_data.electron_wave_rs.getPeriodic(input_wave_window.aa_size[0],
                                                  input_wave_window.aa_size[1],
                                                  input_wave_window.aa_pos[0] - ext_input_wave_window.aa_pos[0],
                                                  input_wave_window.aa_pos[1] - ext_input_wave_window.aa_pos[1]);
  
  return fftw_data.electron_wave_rs.getPeriodic(input_wave_window.aa_size[0],
                                                input_wave_window.aa_size[1],
                                                input_wave_window.aa_pos[0],
                                                input_wave_window.aa_pos[1]);
}

// Implementation of the real space Multislice algorithm avoiding the FFT
Image<complex> MultisliceAlgorithmRS(const AlignedArray& input_wave_window,
                                     const MSData& data,
                                     RealType& norm_change) {
  // Set the initial condition for the electron wave
  Image<complex> electron_wave_rs = data.input.getInitialCondition(input_wave_window);
  
  const RealType input_wave_norm = norm(electron_wave_rs);
  
  for (int i=0; i<static_cast<int>(data.tf.slice_transmission_function.size()); i++) {
    // Transmission step (pointwise multiplication with the transmission function)
    electron_wave_rs *= data.tf.slice_transmission_function[i].getPeriodic(input_wave_window.aa_size[0],
                                                                           input_wave_window.aa_size[1],
                                                                           input_wave_window.aa_pos[0],
                                                                           input_wave_window.aa_pos[1]);
    
    // Propagation step (convolution with the real space propagation function)
    if (data.p.propagator == MS_Propagator::Full) {
      electron_wave_rs = convolve(electron_wave_rs,
                                  data.pf.propagation_function_rs,
                                  {0, 0});
    } else {
      // MS_Propagator::LowRankApproximation
      for (unsigned int i=0; i<data.pf.propagation_function_1D.size(); i++) {
        convolve_rows(electron_wave_rs,
                      data.pf.propagation_function_1D[i],
                      {0, 0});
        convolve_cols(electron_wave_rs,
                      data.pf.propagation_function_1D[i],
                      {0, 0});
      }
    }
  }
  
  // Calculate <norm of exit wave> / <norm of input wave>
  norm_change = norm(electron_wave_rs) / input_wave_norm;
  
  return electron_wave_rs;
}

// Calculates the solution to the Schrödinger equation for the initial guess given by
// data.input centered at input_wave_window.center
//
// Note: the quotient <norm of the exit wave> / <norm of the input wave> is written
//       to norm_change. This value is close to 1 for a successful application of the
//       multislice algorithm
Image<complex> MultisliceAlgorithmCPU(const AlignedArray& input_wave_window,
                                      FFTWData& fftw_data,
                                      const MSData& data,
                                      RealType& norm_change) {
  if (data.p.domain == MS_Domain::FourierSpace)
    return MultisliceAlgorithmFS(input_wave_window, fftw_data, data, norm_change);
  else
    return MultisliceAlgorithmRS(input_wave_window, data, norm_change);
}

#endif  // multislice_cpu_multislicealgorithmcpu_h
