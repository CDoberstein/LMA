// Auxiliary functions for the initialization of the PropagationFunction struct
#ifndef multislice_cpu_propagationfunction_aux_h
#define multislice_cpu_propagationfunction_aux_h

#include "../Utility/Param.h"
#include "../Utility/CircularImage.h"

#include <vector>
    
// Calculates the propagation function in Fourier space as a circular image
//
// Note: fs_pixel_size is the (supersampled) pixel size in Fourier space and not in real
//       space
CircularImage<complex> getPropagationFunctionFS(const Param& p,
                                                const int X,
                                                const RealType fs_pixel_size,
                                                const RealType max_freq) {
  const RealType pi = 3.14159265359;
  
  std::vector<complex> values(X, {0., 0.});
  const int num_val = std::min(X, static_cast<int>(max_freq / fs_pixel_size)+1);
  for (int x=0; x<num_val; x++) {
    const RealType freq = x * fs_pixel_size;
    const RealType arg = -pi * p.lambda * freq * freq * p.slice_thickness;
    
    values[x] = complex(cos(arg), sin(arg));
  }
  
  return CircularImage<complex>(values, fs_pixel_size);
}

#endif  // multislice_cpu_propagationfunction_aux_h
