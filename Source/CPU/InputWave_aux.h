// Auxiliary functions for the initialization of the InputWave struct
#ifndef multislice_cpu_inputwave_aux_h
#define multislice_cpu_inputwave_aux_h

#include "../Utility/Param.h"
#include "../Utility/CircularImage.h"

#include <cmath>
#include <cstdlib>

// Calculates the modulation mu_z(x) = exp(2*Pi*i*(x[0]*z[0] + x[1]*z[1]))
complex modulation(const std::array<RealType, 2> x, const std::array<RealType, 2> z) {
	const RealType pi = 3.14159265359;
	
	const RealType arg = 2 * pi * (x[0] * z[0] + x[1] * z[1]);
	
	return complex(cos(arg), sin(arg));
}

// Evaluates the aberration function for a given frequency norm
// (including only terms for focus and third order spherical aberration)
RealType aberrationFunction(const Param& p, const RealType frequency_norm_squared) {
	const RealType pi = 3.14159265359;
	
	const RealType term1 = p.Z * p.lambda * frequency_norm_squared / 2;
  const RealType term2 = p.Cs * p.lambda * p.lambda * p.lambda * frequency_norm_squared * frequency_norm_squared / 4;
  
  return 2 * pi * (term1 + term2);
}

// Calculates the STEM probe in real space using formula (5.47) on page
// 105 in [Kirkland10]
CircularImage<complex> getProbeKirkland(const Param& p, const RealType pixel_size) {
  // Precompute the probe values up to a distance of three times the given probe radius
  // from the probe position. This is because of LatticeMultisliceAlgorithm, where the
  // propagation window may have a size of up to twice the size of the probe window in both
  // directions. The diagonal consequently has a length of
  //   sqrt(2^2 + 2^2) = sqrt(8) = 2 sqrt(2) = 2.82..
  const RealType ext_probe_radius = p.probe_radius * 3;
  
  // Maximum frequency due to the aperture
  const RealType max_freq = p.alpha_max / p.lambda;
  
  // The probe is calculated with a buffer zone to avoid errors due
  // to the wrap around effect of the FFT. The buffer zone increases the
  // length and width of the image by a factor of (w_factor - 1),
  // where a w_factor of 1 amounts to no buffer zone.
  //
  // Note that this factor needs to be quite large in order to get the same
  // results as with the function getPropagatedProbe() below.
  const RealType w_factor = 2;
  
  // Calculate the Fourier space probe wave function
  const int X = static_cast<int>(std::ceil(2 * w_factor * ext_probe_radius / pixel_size));
  
  Image<complex> probe2D_fs(X, X, {0, 0});
  
  for (int y=0; y<X; y++)
	  for (int x=0; x<X; x++) {
	    const RealType freq_x = (x - X/2) / (2 * w_factor * ext_probe_radius);
	    const RealType freq_y = (y - X/2) / (2 * w_factor * ext_probe_radius);
	    const RealType freq_sqr = freq_x * freq_x + freq_y * freq_y;
	    
	    if (freq_sqr > max_freq * max_freq)
		    continue;
	    
	    const RealType chi = aberrationFunction(p, freq_sqr);
	    
	    probe2D_fs(x, y) = complex(cos(chi), -sin(chi));
	  }
  
  // Calculate the real space probe wave function
  Image<complex> probe2D_rs(X, X, {0, 0});
  
  probe2D_fs.applyInverseFourierShift();
  InvFourierTransform(probe2D_fs, &probe2D_rs);
  
  // Normalize so that the 2-Norm equals 1
  probe2D_rs /= norm(probe2D_rs);
  
  // Convert to a circular image
  std::vector<complex> line(X / static_cast<int>(2 * w_factor));
  for (int i=0; i<static_cast<int>(line.size()); i++)
	line[i] = probe2D_rs(i, 0);
  
  return CircularImage<complex>(line, pixel_size);
}

// Calculates the STEM probe in real space after propagation through l
// empty slices with thickness given by p.slice_thickness.
// The STEM probe is calculated directly in real space using a formula that
// can be derived from the standard formula by using the definition of the
// Fourier transform and integrating parts of the integrals analytically.
CircularImage<complex> getPropagatedProbe(const Param& p, const RealType pixel_size, const int l) {
  const RealType pi = 3.14159265359;
  
  // Precompute the probe values up to a distance of three times the given probe radius
  // from the probe position. This is because of LatticeMultisliceAlgorithm, where the
  // propagation window may have a size of up to twice the size of the probe window in both
  // directions. The diagonal consequently has a length of
  //   sqrt(2^2 + 2^2) = sqrt(8) = 2 sqrt(2) = 2.82..
  const RealType ext_probe_radius = p.probe_radius * 3;
  
  // Set the Fourier space sampling size eps based on the distance of successive zeros of
  //   r -> J_0(2*pi*rho*r),
  // which is approximately 1 / (2 * rho) since the distance of successive zeros of the
  // Bessel functions of the first kind is approximately equal to pi.
  const RealType eps = (1 / (2 * ext_probe_radius)) / 100;
  
  // Upper bound of the integration: maximum frequency due to the aperture
  const RealType max_freq = p.alpha_max / p.lambda;
  
  // Total number of samples for one evaluation of the integral
  const int num_int_samples = static_cast<int>(std::ceil(max_freq / eps));
  
  // Precompute the values of the r * exp(-i*(chi(r) + l * pi * lambda * t * r * r))
  // factor of the integrand
  std::vector<complex> f1(num_int_samples);
  for (int i=0; i<num_int_samples; i++) {
    const RealType freq = i * eps;
    
    const RealType chi = aberrationFunction(p, freq*freq);
    const RealType arg = -chi - l * pi * p.lambda * p.slice_thickness * freq * freq;
    
    f1[i] = complex(freq * cos(arg), freq * sin(arg));
  }
  
  // Perform the integration for every pixel of the resulting circular probe image
  const int num_probe_samples = static_cast<int>(std::ceil(ext_probe_radius / pixel_size));
  
  std::vector<complex> line(num_probe_samples);
  for (int i=0; i<num_probe_samples; i++) {
    // Distance to the center of the probe
    const RealType dist = i * pixel_size;
    
    complex integral = {0., 0.};
    for (int j=0; j<num_int_samples; j++) {
      const RealType freq = j * eps;
      const RealType f2 = std::cyl_bessel_j(0, 2 * pi * freq * dist);
      
      integral += f1[j] * f2;
    }
    integral *= 2 * pi * eps;
    
    line[i] = integral;
  }
  
  return CircularImage<complex>(line, pixel_size);
}

// Calculates a X by X pixel image of a trigonometric polynomial of given
// degree and pixel size, where the origin is located at (X/2, X/2)
Image<complex> trigPoly(const int X,
                        const int degree,
                        const RealType pixel_size) {
  // Auxiliary function for the computation of the trigonometric polynomials
  auto phi = [degree](const RealType t) {
    const RealType pi = 3.14159265359;
    
    complex res = {0, 0};
    for (int k=-degree; k<=degree; k++) {
      const RealType coeff = cos(k*pi/(2*degree+2));
      res += complex(cos(k*t) * coeff, sin(k*t) * coeff);
    }
    
    return res;
  };
  
  // Calculate an image of size X times X of the trigonometric polynomial
  Image<complex> result(X, X, {0, 0});
  for (int y=0; y<X; y++)
  	for (int x=0; x<X; x++) {
	  const RealType dx = (x - X/2) * pixel_size;
	  const RealType dy = (y - X/2) * pixel_size;
	  
	  result(x, y) = phi(dx) * phi(dy);
	}
  
  return result;
}

// Calculate a trigonometric polynomial function as described in
//
//   Holger Rauhut, Best time localized trigonometric polynomials and wavelets.
//   Advances in Computational Mathematics (2005) 22: 1–20
//
Image<complex> getTrigPolyFunction(const Param& p,
                                   const RealType pixel_size,
                                   RealType& radius) {
  const RealType pi = 3.14159265359;
  
  // The image size is limited by the 2*pi periodicity of phi (see trigPoly() above)
  const int X = 2 * pi / pixel_size;
  
  // Calculate the degree of the trigonometric polynomials based on
  // the maximum probe frequency if p.trig_poly_degree == -1
  const RealType max_freq = p.alpha_max / p.lambda;
  const RealType fs_pixel_size = 1 / (pixel_size * X);
  const int degree = (p.trig_poly_degree < 0 ? std::ceil(max_freq / fs_pixel_size)
                                             : p.trig_poly_degree);
  
  if (degree <= 10)
    radius = pi;
  else
    radius = 10 * (pi / degree);
  
  std::cerr << "\tTrigonometric polynomial degree: " << degree << std::endl
            << "\tMinimum radius: " << radius << " Angstrom" << std::endl;
  
  // Calculate a 2D image of a trigonometric polynomial in real space
  Image<complex> poly2D_rs = trigPoly(X, degree, pixel_size);
  
  // Crop image slightly to ensure that the values near the image edges
  // are close to zero in order to reduce artifacts when sampling as a
  // part of a bigger image
  int crop_left = 0;
  for (int x=0; x<X/2; x++) {
    const RealType v1 = poly2D_rs(x, X/2).real();
    const RealType v2 = poly2D_rs(x+1, X/2).real();
    
    // Search for the first intersection with the x axis, if any
    if ((v1 <= 0 && v2 >= 0) || (v1 >= 0 && v2 <= 0)) {
      if (std::abs(v1) < std::abs(v2))
		crop_left = x;
      else
		crop_left = x+1;
	  
	  break;
    }
  }
  
  int crop_right = X-1;
  for (int x=X-1; x>=X/2; x--) {
    const RealType v1 = poly2D_rs(x, X/2).real();
    const RealType v2 = poly2D_rs(x-1, X/2).real();
    
    // Search for the first intersection with the x axis, if any
    if ((v1 <= 0 && v2 >= 0) || (v1 >= 0 && v2 <= 0)) {
      if (std::abs(v1) < std::abs(v2))
		crop_right = x;
      else
		crop_right = x-1;
	  
	  break;
    }
  }
  
  poly2D_rs = poly2D_rs.crop(crop_right - crop_left + 1,
                             crop_right - crop_left + 1,
                             crop_left,
                             crop_left);
  
  // Normalize so that the 2-Norm equals 1
  poly2D_rs /= norm(poly2D_rs);
  
  return poly2D_rs;
}

// Similar to getTrigPolyFunction() above, but the Fourier space frequencies
// of the trigonometric polynomial are cropped to the frequencies of
// the probe wave function (i.e. the frequencies within the objective aperture)
CircularImage<complex> getTrigPolyFunctionCroppedFS(const Param& p,
                                                    const RealType pixel_size,
                                                    RealType& radius) {
  const RealType pi = 3.14159265359;
  
  // The image size is limited by the 2*pi periodicity of phi (see trigPoly() above)
  const int X = 2 * pi / pixel_size;
  
  // Calculate the degree of the trigonometric polynomials based on
  // the maximum probe frequency if p.trig_poly_degree == -1
  const RealType max_freq = p.alpha_max / p.lambda;
  const RealType fs_pixel_size = 1 / (pixel_size * X);
  const int degree = (p.trig_poly_degree < 0 ? std::ceil(max_freq / fs_pixel_size)
                                             : p.trig_poly_degree);
  
  if (degree <= 15)
    radius = pi;
  else
    radius = 15 * (pi / degree);
  
  std::cerr << "\tTrigonometric polynomial degree: " << degree << std::endl
            << "\tMinimum radius: " << radius << " Angstrom" << std::endl;
  
  // Calculate a 2D image of a trigonometric polynomial in real space
  Image<complex> poly2D_rs = trigPoly(X, degree, pixel_size);
  
  // Crop the frequencies to the aperture radius
  Image<complex> poly2D_fs(poly2D_rs);
  FourierTransform(poly2D_rs, &poly2D_fs);
  poly2D_fs.applyFourierShift();
  
  for (int y=0; y<X; y++)
    for (int x=0; x<X; x++) {
      const RealType freq_x = (x - X/2) / (X * pixel_size);
      const RealType freq_y = (y - X/2) / (X * pixel_size);
      const RealType freq_sqr = freq_x * freq_x + freq_y * freq_y;
      
      if (freq_sqr > max_freq * max_freq)
        poly2D_fs(x, y) = complex(0, 0);
    }
  
  poly2D_fs.applyInverseFourierShift();
  InvFourierTransform(poly2D_fs, &poly2D_rs);
  poly2D_rs /= X * X;
  
  // Normalize
  poly2D_rs /= norm(poly2D_rs);
  
  // Search for the last sign change on the line with y = X/2
  int max_x = X;
  for (int x=X-1; x>X/2; x--) {
    const RealType v1 = poly2D_rs(x, X/2).real();
    const RealType v2 = poly2D_rs(x-1, X/2).real();
    
    if ((v1 <= 0 && v2 >= 0) || (v1 >= 0 && v2 <= 0)) {
      if (std::abs(v1) < std::abs(v2))
		max_x = x+1;
	  else
	    max_x = x;
	  
	  break;
	}
  }
  
  // Convert to a circular image
  std::vector<complex> line(max_x - X/2);
  for (int i=X/2; i<max_x; i++)
    line[i-X/2] = poly2D_rs(i, X/2);
  
  return CircularImage<complex>(line, pixel_size);
}

// Calculates an image of the indicator function of a square
Image<complex> getSquareInputWave(const Param& p,
                                  const RealType pixel_size) {
  const int X = p.input_wave_square_len / pixel_size;
  const RealType len = X * pixel_size;
  
  std::cerr << "\tSquare input wave size: " << len << " x " << len << " (Angstrom)" << std::endl
            << "\t                        " << X << " x " << X << " (Pixel)" << std::endl;
  
  return Image<complex>(X, X, {1, 0});
}

// Calculates an image of the indicator function of a disk
CircularImage<complex> getDiskInputWave(const Param& p,
                                        const RealType pixel_size) {
  const int X = p.input_wave_disk_radius / pixel_size;
  const RealType radius = X * pixel_size;
  
  std::cerr << "\tDisk input wave radius: " << radius << " (Angstrom)" << std::endl
            << "\t                        " << X << " (Pixel)" << std::endl;
  
  return CircularImage<complex>(std::vector<complex>(X, {1, 0}), pixel_size);
}

// Returns the input wave that is constant zero except at the pixel
// given by (pos_x, pos_y), where it is equal to 1
Image<complex> getPixelInputWave(const RealType sampling_pixel_size_x,
                                 const RealType sampling_pixel_size_y,
                                 const int X,
                                 const int Y,
                                 const RealType pos_x,
                                 const RealType pos_y) {
  Image<complex> result(X, Y, {0, 0});
  
  int x = normalize_periodic(static_cast<int>(std::floor(pos_x / sampling_pixel_size_x)), X);
  int y = normalize_periodic(static_cast<int>(std::floor(pos_y / sampling_pixel_size_y)), Y);
  
  result(x, y) = {1, 0};
  
  return result;
}

// Calculates an image of a gaussian with standard deviation p.input_wave_gaussian_sigma
CircularImage<complex> getGaussianInputWave(const Param& p,
                                            const RealType pixel_size,
                                            RealType& radius) {
  // Set the standard deviation in Angstrom
  RealType sigma;
  if (p.input_wave_gaussian_sigma <= 0) {
    // Calculate a reasonable value for sigma based on the Fourier space
    // frequencies in the probe wave function (i.e. the objective aperture
    // radius)
    const RealType max_freq = p.alpha_max / p.lambda;
    
    sigma = 1 / (2 * max_freq);
  } else
	sigma = p.input_wave_gaussian_sigma;
  
  radius = 3 * sigma;
  std::cerr << "\tGaussian input wave sigma: " << sigma << " (Angstrom)" << std::endl;
  
  // Calculate a circular image
  const int X = 5 * sigma / pixel_size;
  std::vector<complex> line(X);
  
  for (int i=0; i<X; i++) {
    const RealType x = i * pixel_size;
    
    line[i] = complex(std::exp(-x*x/(2*sigma*sigma)), 0);
  }
  
  return CircularImage<complex>(line, pixel_size);
}

// Returns the input wave in real space corresponding a Dirac delta at
// the pixel coordinate (x, y) in Fourier space
Image<complex> getFSDiracInputWave(const int X,
                                   const int Y,
                                   const int x,
                                   const int y) {
  Image<complex> result_fs(X, Y, {0, 0});
  
  result_fs(X/2 + x, Y/2 + y) = {1, 0};
  result_fs.applyInverseFourierShift();
  
  Image<complex> result_rs(X, Y, {0, 0});
  InvFourierTransform(result_fs, &result_rs);
  
  return result_rs;
}

#endif  // multislice_cpu_inputwave_aux_h
