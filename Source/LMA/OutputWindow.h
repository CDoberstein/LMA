// Container for the dimensions of the 3D STEM output
#ifndef multislice_outputwindow_h
#define multislice_outputwindow_h

#include "../Utility/Param.h"
#include "../Utility/Lattice.h"
#include "../CPU/PropagationWindow.h"

#include <algorithm>
#include <cmath>

struct OutputWindow {
  // Size of the resulting 3D STEM image in pixel
  //
  // Note: Z is the number of annulus detectors calculated from the detector_stepsize
  //       parameter and the magnitude of the highest frequency in the simulated images
  int X, Y, Z;
  
  // Index of the top left probe position of the output window (with respect to the top
  // left probe position in the probe lattice)
  int start_x, start_y;
  
  OutputWindow() = default;
  
  OutputWindow(const Param& p,
               const Lattice& probe_lattice,
               const PropagationWindow& propagation_window) {
    // Size of the output image in pixels
    X = static_cast<int>(std::ceil(probe_lattice.X * (p.simulation_bounds_x[1] - p.simulation_bounds_x[0])));
    Y = static_cast<int>(std::ceil(probe_lattice.Y * (p.simulation_bounds_y[1] - p.simulation_bounds_y[0])));
    
    X = std::clamp(X, 1, probe_lattice.X);
    Y = std::clamp(Y, 1, probe_lattice.Y);
    
    // Calculate the maximum scattering angle according to the propagation window pixel
    // size (in mrad)
    //
    // Note: the maximum frequency is divided by 4 instead of 2 because of the
    //       antialiasing aperture in the propagation function
    const RealType max_freq_x = 1 / (4 * propagation_window.pixel_size_x);
    const RealType max_freq_y = 1 / (4 * propagation_window.pixel_size_y);
    
    const RealType max_scattering_angle = p.lambda * std::min(max_freq_x, max_freq_y) * 1000;
    
    // Number of annulus detectors
    Z = max_scattering_angle / p.detector_stepsize;
    
    Z = std::max(1, Z);
    
    // The index of the first (top left) probe position in the output window
    start_x = static_cast<int>(p.simulation_bounds_x[0] * probe_lattice.X);
    start_y = static_cast<int>(p.simulation_bounds_y[0] * probe_lattice.Y);
  }
};

#endif  // multislice_outputwindow_h
