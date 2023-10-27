// Container for the dimensions of the propagation window used for transmitting the
// input waves through the specimen in the real space algorithms (MS_Domain::RealSpace)
#ifndef multislice_cpu_propagationwindow_h
#define multislice_cpu_propagationwindow_h

#include "SimulationWindow.h"
#include "InputWave.h"

#include "../Utility/Param.h"

#include <algorithm>
#include <cmath>

struct PropagationWindow {
  // Size of the propagation window in pixels
  int X, Y;
  
  // Size of the propagation window in Angstrom
  RealType lenX, lenY;
  
  // Size of one pixel of the propagation window in Angstrom
  RealType pixel_size_x, pixel_size_y;
  
  // Size of the propagation window in pixels including the buffer zone
  // according to p.fs_buffer_zone
  int ext_X, ext_Y;
  
  // Size of the propagation window in Angstrom including the buffer zone
  // according to p.fs_buffer_zone
  RealType ext_lenX, ext_lenY;
  
  void init(const Param& p,
            const InputWave& input_wave,
            const SimulationWindow& simulation_window) {
    // The propagation window size must be at least twice the input wave
    // radius, but there is no advantage in having a propagation window
    // size greater than twice the probe window size. This is because
    // the results of the Multislice algorithm are cropped to the probe
    // window.
    if (input_wave.initial_condition_radius > 2 * p.probe_radius)
      lenX = 2 * input_wave.initial_condition_radius;
    else
      lenX = std::clamp(p.propagation_window_size,
                        2 * input_wave.initial_condition_radius,
                        4 * p.probe_radius);
    
    lenY = lenX;
    
    // Use the same pixel sizes as for the simulation window to ensure compatibility when
    // using the Fourier space algorithms, because the resulting exit waves (which are
    // calculated using SimulationWindow and not PropagationWindow in the Fourier space
    // algorithms) are cropped to the propagation window to save memory.
    pixel_size_x = simulation_window.pixel_size_x;
    pixel_size_y = simulation_window.pixel_size_y;
    
    // The propagation window is forced to have an even number of pixels in both
    // directions
    X = 2 * static_cast<int>(std::ceil((lenX / pixel_size_x) / 2));
    Y = 2 * static_cast<int>(std::ceil((lenY / pixel_size_y) / 2));
      
    // Adjust the side lengths of the propagation window
    lenX = X * pixel_size_x;
    lenY = Y * pixel_size_y;
    
    // Set the sizes for the extended propagation window including a buffer zone
    if (p.fs_buffer_zone >= 0) {
			ext_X = static_cast<int>(X * (1 + 2 * p.fs_buffer_zone));
			ext_Y = static_cast<int>(Y * (1 + 2 * p.fs_buffer_zone));
			
			ext_lenX = ext_X * pixel_size_x;
			ext_lenY = ext_Y * pixel_size_y;
			
			if (p.domain == MS_Domain::FourierSpace)
			  if (ext_X > simulation_window.X || ext_Y > simulation_window.Y)
			    Error("The extended propagation window size (" + std::to_string(ext_X) + ", "
			          + std::to_string(ext_Y) + ") exceeds the simulation window size ("
			          + std::to_string(simulation_window.X) + ", "
			          + std::to_string(simulation_window.Y) + "). The value of the "
			          "\"fs_buffer_zone\" parameter must be decreased or set to -1.");
			
			if (p.inputwave == MS_InputWave::FourierSpaceDirac)
			  Error("The parameter \"fs_buffer_zone\" must be set to a negative value "
			        "when using the FourierSpaceDirac input wave type!");
		} else {
			ext_X = X;
			ext_Y = Y;
			
			ext_lenX = lenX;
			ext_lenY = lenY;
		}
			
  }
};

#endif  // multislice_cpu_propagationwindow_h
