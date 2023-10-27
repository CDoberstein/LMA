#ifndef multislice_cpu_propagationfunction_h
#define multislice_cpu_propagationfunction_h

#include "PropagationFunction_aux.h"

#include "../Utility/CircularImage.h"
#include "../Utility/Image.h"
#include "../Utility/Param.h"
#include "SimulationWindow.h"

#include <Eigen/Core>
#include <Eigen/SVD>

#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>

struct PropagationFunction {
  // Fourier space propagation function matching the size of the full simulation window
  // (used if p.domain == MS_Domain::FourierSpace)
  Image<complex> propagation_function_fs;
  
  // Real space propagation function kernel (used if p.domain == MS_Domain::RealSpace)
  Image<complex> propagation_function_rs;
  
  // Decomposition of the real space propagation function into a sum of rank 1 matrices
  // given by the outer products of propagation_function_1D[i] with themselves
  //
  // Note: this is only computed if p.propagator is MS_Propagator::LowRankApproximation
  std::vector<std::vector<complex>> propagation_function_1D;
  
  void init(const Param& p,
            const SimulationWindow& simulation_window,
            const PropagationWindow& propagation_window) {
    // Calculate the Fourier space propagation function
    //
    // Note: the antialiasing aperture is included by dividing the maximum frequency, i.e.
    //       the last argument to getPropagationFunctionFS, by 2. This is done for both
    //       the real space and the Fourier space propagation function.
    std::cerr << "\tCalculating the Fourier space propagation function ..." << std::endl;
    CircularImage<complex> pf_circular;
    pf_circular = getPropagationFunctionFS(p,
                                           p.hires_supersampling_factor * std::max(simulation_window.X, simulation_window.Y),
                                           1 / (p.hires_supersampling_factor * std::max(simulation_window.lenX, simulation_window.lenY)),
                                           std::min(simulation_window.X / (2 * simulation_window.lenX), simulation_window.Y / (2 * simulation_window.lenY)) / 2);
    
    if (p.fs_buffer_zone >= 0) {
			// The Fourier space Multislice algorithm is performed on the
			// propagation window extended by a buffer zone
			propagation_function_fs = pf_circular.get2DImageNonSquare(1/propagation_window.ext_lenX,
			                                                          1/propagation_window.ext_lenY,
			                                                          propagation_window.ext_X,
			                                                          propagation_window.ext_Y,
			                                                          {0, 0});
		} else {
			// The Fourier space Multislice algorithm is performed on the
			// full simulation window
			propagation_function_fs = pf_circular.get2DImageNonSquare(1/simulation_window.lenX,
																																1/simulation_window.lenY,
																																simulation_window.X,
																																simulation_window.Y,
																																{0, 0});
    }
    
    if (p.writeInitData)
      save(propagation_function_fs, "PropagationFunction_FS", p.outputDir + "/MSData/PropagationFunction");
    
    // Calculate the propagation function in real space ...
    std::cerr << "\tCalculating the real space propagation function ..." << std::endl;
    
    auto ceil_odd_int = [](const RealType v) {
      int res = static_cast<int>(std::ceil(v));
      return res + (res+1) % 2;
    };
    
    const int X = ceil_odd_int(p.min_propagator_radius / propagation_window.pixel_size_x);
    const int Y = ceil_odd_int(p.min_propagator_radius / propagation_window.pixel_size_y);
    
    const RealType propagator_radius_x = X * propagation_window.pixel_size_x;
    const RealType propagator_radius_y = Y * propagation_window.pixel_size_y;
    
    std::cerr << "\t\tSize: " << X << " x " << Y << " (pixel)" << std::endl
              << "\t\t      " << propagator_radius_x << " x " << propagator_radius_y << " (Angstrom)" << std::endl;
    
    Image<complex> tmp = pf_circular.get2DImageNonSquare(1 / propagator_radius_x,
                                                         1 / propagator_radius_y,
                                                         X,
                                                         Y,
                                                         {0, 0});
    tmp.applyInverseFourierShift();
    propagation_function_rs = Image<complex>(X, Y, {0, 0});
    InvFourierTransform(tmp, &propagation_function_rs);
    propagation_function_rs.applyFourierShift();
    propagation_function_rs /= static_cast<RealType>(X) * Y;
    
    // ... and ensure that the entries sum to 1 + 0 * I
    const complex add = (complex{1, 0} - sum(propagation_function_rs)) / complex(X * Y, 0);
    for (int j=0; j<X*Y; j++)
      propagation_function_rs[j] += add;
    
    if (p.writeInitData) {
      save(propagation_function_rs,
           "PropagationFunction_RS",
           p.outputDir + "/MSData/PropagationFunction");
      
      save(pointwise_norm(propagation_function_rs),
           "PropagationFunction_RS_norm",
           p.outputDir + "/MSData/PropagationFunction");
    } 
    
    // Compute a decomposition of the real space propagation function as a sum of rank 1
    // matrices
    if (p.propagator == MS_Propagator::LowRankApproximation) {
      if (propagation_window.pixel_size_x != propagation_window.pixel_size_y)
        Error("The low rank approximation of the propagation function is not implemented "
              " for non-square simulation pixels"
              " (propagation_window.pixel_size_x != propagation_window.pixel_size_y)");
      
      std::cerr << "\tCalculating a rank-1 decomposition of the real space propagation function ..." << std::endl;
      
      // propagation_function_rs is a symmetric matrix, so the SVD yields a decomposition
      // with the desired properties
      Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> A(X, X);

      for (int x=0; x<X; x++)
        for (int y=0; y<X; y++)
          A.coeffRef(x, y) = propagation_function_rs(x, y);
      
      Eigen::BDCSVD svd(A, Eigen::ComputeFullU);
      auto Sigma = svd.singularValues();
      auto U = svd.matrixU();
      
      int c = 0;
      for (; c<X; c++) {
        if (Sigma[c] == 0 || c == p.propagator_rank)
          break;
        
        std::vector<complex> singular_vector(X, {0, 0});
        for (int r=0; r<X; r++)
          singular_vector[r] = U(r, c);
          
        propagation_function_1D.push_back(sqrt(Sigma[c]) * singular_vector);
      }
      
      std::cerr << "\t\tNumber of rank-1 kernels: " << c << std::endl;
      
      if (p.writeInitData) {
        Image<complex> approximation(X, X, {0, 0});
        std::vector<RealType> residuals;
        for (unsigned int i=0; i<propagation_function_1D.size(); i++) {
          // Add the outer product of propagation_function_1D[i] with itself to approximation
          for (int y=0; y<X; y++)
            for (int x=0; x<X; x++)
              approximation(x, y) += propagation_function_1D[i][x] * propagation_function_1D[i][y];
          
          save(approximation,
               std::to_string(i+1),
               p.outputDir + "/MSData/PropagationFunction/Approximation/Re",
               p.outputDir + "/MSData/PropagationFunction/Approximation/Im");
          
          Image<complex> diff(propagation_function_rs);
          diff -= approximation;
          residuals.push_back(norm(diff));
        }
        
        std::ofstream fres(p.outputDir + "/MSData/PropagationFunction/Approximation/residuals");
        if (fres.is_open())
          for (unsigned int i=0; i<residuals.size(); i++)
            fres << std::setw(5) << ": " << residuals[i] << std::endl;
      }
    }
    
    if (p.propagator == MS_Propagator::LowRankApproximation && p.domain == MS_Domain::FourierSpace)
      std::cerr << "\tWARNING: \"LowRankApproximation\" can not be used with the Fourier space" << std::endl
                << "\t         Multislice algorithm. Using the standard propagation function instead." << std::endl;
  }
};

#endif  // multislice_cpu_propagationfunction_h
