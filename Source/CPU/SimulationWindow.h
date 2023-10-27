// Container for the dimensions of the full simulation window covering the whole specimen,
// which is used in the Fourier space algorithms (MS_Domain::FourierSpace)
#ifndef multislice_cpu_simulationwindow_h
#define multislice_cpu_simulationwindow_h

#include "../Utility/Param.h"
#include "../Utility/Specimen.h"

#include <cmath>

struct SimulationWindow {
  // Size of the full simulation window in pixel
  int X, Y;
  
  // Width and height of the simulation window in Angstrom (= width and height of the
  // specimen)
  RealType lenX, lenY;
  
  // Simulation pixel size
  //
  // Note: this may be slightly different from the size given by the parameters
  //       p.req_simulation_pixel_size_{x/y} to ensure that the width and height of the
  //       simulation window is an even integer multiple of the simulation pixel size
  RealType pixel_size_x;
  RealType pixel_size_y;
  
  void init(const Param& p, const Specimen& specimen) {
    // Auxiliary function to round up to the next even integer
    auto round_to_even = [](const RealType value) {
      int res = std::ceil(value);
      return (res % 2 == 1 ? res + 1 : res);
    };
    
    // Compute the size of the simulation window in pixel
    X = round_to_even(specimen.lenX / p.req_simulation_pixel_size_x);
    Y = round_to_even(specimen.lenY / p.req_simulation_pixel_size_y);
    
    // The width and height of the simulation window is equal to the size of the specimen
    lenX = specimen.lenX;
    lenY = specimen.lenY;
    
    // The actual simulation pixel size in Angstrom
    pixel_size_x = lenX / X;
    pixel_size_y = lenY / Y;
  }
};

#endif  // multislice_cpu_simulationwindow_h
