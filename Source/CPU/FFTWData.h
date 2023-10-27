#ifndef multislice_cpu_fftwdata_h
#define multislice_cpu_fftwdata_h

#include "../Utility/Image.h"

#include <fftw3.h>

// Preallocated arrays and plans for using FFTW in MultisliceAlgorithmFS()
struct FFTWData {
  private:
    // Memory locations as passed to the FFTW plan generation routines
    complex *rs_addr;
    complex *fs_addr;
  
  public:
    Image<complex> electron_wave_rs;  // rs = real space
    Image<complex> electron_wave_fs;  // fs = Fourier space
    
    fftw_plan forward_transform;
    fftw_plan backward_transform;
    
    void init(const int X, const int Y) {
      electron_wave_rs = Image<complex>(X, Y, complex(0, 0));
      electron_wave_fs = Image<complex>(X, Y, complex(0, 0));
      
      rs_addr = &electron_wave_rs[0];
      fs_addr = &electron_wave_fs[0];
      
      forward_transform = fftw_plan_dft_2d(Y,
                                           X,
                                           reinterpret_cast<fftw_complex*>(rs_addr),
                                           reinterpret_cast<fftw_complex*>(fs_addr),
                                           FFTW_FORWARD,
                                           FFTW_PATIENT);
      
      backward_transform = fftw_plan_dft_2d(Y,
                                            X,
                                            reinterpret_cast<fftw_complex*>(fs_addr),
                                            reinterpret_cast<fftw_complex*>(rs_addr),
                                            FFTW_BACKWARD,
                                            FFTW_PATIENT);
    }
    
    void cleanup() {
      fftw_destroy_plan(forward_transform);
      fftw_destroy_plan(backward_transform);
    }
    
    // If this function returns false, the forward and backward transform
    // in this instance must not be used anymore to prevent memory errors
    //
    // Note: this may be caused by some functions of the Image class
    //       that modify the underlying data vector
    //
    // Note 2: this should really be a const function, but C++ doesn't
    //         allow taking addresses of rvalues
    bool validMemAddr() {
      return (rs_addr == &electron_wave_rs[0] && fs_addr == &electron_wave_fs[0]);
    }
};

#endif  // multislice_cpu_fftwdata_h
