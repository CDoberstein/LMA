// Implementation of a container class for 2D images
#ifndef multislice_utility_image_h
#define multislice_utility_image_h

#include "Error.h"
#include "complex.h"

#include <netcdf.h>

#include <omp.h>
#include <fftw3.h>

#include <iostream>
#include <vector>
#include <array>
#include <string>
#include <utility>
#include <algorithm>
#include <filesystem>
#include <cmath>
#include <ostream>
#include <new>
#include <memory>

// Custom allocator class for images with data type complex to use 
// fftw_alloc_complex with appropriate memory alignment for FFTW
//
// Note: the only valid type for T is complex.
template <typename T>
struct FFTW_Allocator {
  typedef T value_type;
  
  FFTW_Allocator() = default;
  constexpr FFTW_Allocator(const FFTW_Allocator&) noexcept {}
  
  T* allocate(std::size_t n) {
	if (fftw_complex *p = fftw_alloc_complex(n))
	  return reinterpret_cast<T*>(p);
	
	throw std::bad_alloc();
  }
  
  void deallocate(T *p, std::size_t n) noexcept {
    fftw_free(reinterpret_cast<fftw_complex*>(p));
  }
};

template<class T, class U>
bool operator==(const FFTW_Allocator <T>&, const FFTW_Allocator <U>&) { return true; }
 
template<class T, class U>
bool operator!=(const FFTW_Allocator <T>&, const FFTW_Allocator <U>&) { return false; }


template <typename T> class Image;

void impl_FourierTransform(Image<complex>&, Image<complex>*, bool);
void save(const Image<complex>&, const std::string&, const std::string&, const std::string&);
void save(const Image<RealType>&, const std::string&, const std::string&);

// 2D image class
template <typename T>
class Image {
  private:
    // Image dimensions
    int X;
    int Y;
    
    // Pixel values, using the FFTW allocator if the data type is complex
    typedef std::conditional_t< std::is_same<T, complex>::value , FFTW_Allocator<complex>, std::allocator<T> > Allocator;
    
    std::vector<T, Allocator> data;
  
  public:
    Image() : X(0), Y(0) {}
    Image(int X, int Y, T init_val) : X(X), Y(Y), data(X*Y, init_val) {}
    
    T operator()(const int x, const int y) const { return data[x + y*X]; }
    T& operator()(const int x, const int y) { return data[x + y*X]; }
    
    T operator()(const std::array<int, 2>& p) const { return data[p[0] + p[1]*X]; }
    T& operator()(const std::array<int, 2>& p) { return data[p[0] + p[1]*X]; }
    
    void set(const std::array<int, 2>& p, const T& value) { data[p[0] + p[1]*X] = value; }
    T get(const std::array<int, 2>& p) const { return data[p[0] + p[1]*X]; }
    
    T operator[](const int i) const { return data[i]; }
    T& operator[](const int i) { return data[i]; }
    
    int getX() const { return X; }
    int getY() const { return Y; }
    
    int size() const { return static_cast<int>(data.size()); }
    
    bool empty() const { return size() == 0; }
    
    // Custom operator= that avoids an address change of data[0] if the
    // images are of the same size (this is important for performing the
    // DFT with precomputed FFTW plans)
    Image& operator=(const Image& rhs) {
      if (X == rhs.X && Y == rhs.Y) {
		for (unsigned int i=0; i<data.size(); i++)
		  data[i] = rhs.data[i];
	  } else {
	    X = rhs.X;
	    Y = rhs.Y;
	    
	    // Force freeing memory if the new Image size is zero
	    if (X == 0 && Y == 0)
		  std::vector<T, Allocator>().swap(data);
		else
	      data = rhs.data;
	  }
	  
	  return *this;
    }
    
    // Apply a Fourier shift to the image
    //
    // Important: Applying the Fourier shift twice to images with an odd number of pixels
    //            in X or Y direction does NOT return the original image (see also
    //            applyInverseFourierShift() below)
    void applyFourierShift() {
      const int Xh = X / 2;
      const int Yh = Y / 2;
      
      if (X % 2 != 0 || Y % 2 != 0) {
        // Odd number of pixels in X or Y direction
        std::vector<T> result(data.size());
        
        for (int y=0; y<Y; y++)
          for (int x=0; x<X; x++) {
            const int res_x = (x + Xh) % X;
            const int res_y = (y + Yh) % Y;
            
            result[res_x + res_y*X] = data[x + y*X];
          }
        
        // Don't use "data = result" to avoid an address change of data[0]
        for (unsigned int i=0; i<data.size(); i++)
		  data[i] = result[i];
      } else {
        // Even number of pixels in both X and Y direction
        auto simple_swap = [](T& a, T& b) { T val = a; a = b; b = val; };
        
        for (int y=0; y<Yh; y++)
          for (int x=0; x<Xh; x++) {
            simple_swap(data[x + y*X], data[x+Xh + (y+Yh)*X]);
            simple_swap(data[x+Xh + y*X], data[x + (y+Yh)*X]);
          }
      }
    }
    
    // Apply an inverse Fourier shift to the image
    void applyInverseFourierShift() {
      if (X % 2 != 0 || Y % 2 != 0) {
        // Odd number of pixels in X or Y direction
        const int Xh = X / 2 + (X % 2 == 0 ? 0 : 1);
        const int Yh = Y / 2 + (Y % 2 == 0 ? 0 : 1);
        
        std::vector<T> result(data.size());
        
        for (int y=0; y<Y; y++)
          for (int x=0; x<X; x++) {
            const int res_x = (x + Xh) % X;
            const int res_y = (y + Yh) % Y;
            
            result[res_x + res_y*X] = data[x + y*X];
          }
        
        // Don't use "data = result" to avoid an address change of data[0]
        for (unsigned int i=0; i<data.size(); i++)
		  data[i] = result[i];
      } else {
        const int Xh = X / 2;
        const int Yh = Y / 2;
        
        // Even number of pixels in both X and Y direction
        auto simple_swap = [](T& a, T& b) { T val = a; a = b; b = val; };
        
        for (int y=0; y<Yh; y++)
          for (int x=0; x<Xh; x++) {
            simple_swap(data[x + y*X], data[x+Xh + (y+Yh)*X]);
            simple_swap(data[x+Xh + y*X], data[x + (y+Yh)*X]);
          }
      }
    }
    
    // Returns a subsection of the image
    Image<T> crop(int nX, int nY, int offsetX, int offsetY) const {
      offsetX = std::clamp(offsetX, 0, X);
      offsetY = std::clamp(offsetY, 0, Y);
      
      nX = std::clamp(nX, 0, X - offsetX);
      nY = std::clamp(nY, 0, Y - offsetY);
      
      Image<T> res(nX, nY, data[0]);
      
      for (int y=0; y<nY; y++)
        for (int x=0; x<nX; x++)
          res(x, y) = this->operator()(x+offsetX, y+offsetY);
      
      return res;
    }
    
    // Returns a (sub)section of the image of the size nX x nY with top left corner at
    // (x0, y0) assuming periodic continuation beyond the image boundaries
    Image<T> getPeriodic(int nX, int nY, int x0, int y0) const {
      if (x0 < 0)
        x0 += (-x0 / X + 1) * X;
      
      if (y0 < 0)
        y0 += (-y0 / Y + 1) * Y;
      
      Image<T> res(nX, nY, data[0]);
      
      for (int y=0; y<nY; y++)
        for (int x=0; x<nX; x++)
          res(x, y) = this->operator()((x0 + x) % X, (y0 + y) % Y);
      
      return res;
    }
    
    // Bilinear interpolation with constant continuation by ext_value
    T interpolate(const RealType x, const RealType y, const T ext_value) const {
      const int x_int = x;
      const int y_int = y;
      
      const RealType x_frac = x - x_int;
      const RealType y_frac = y - y_int;
      
      auto get_val = [&](const int x, const int y) {
        if (x>=0 && x<X && y>=0 && y<Y)
		  return this->operator()(x, y);
		return ext_value;
      };
      
      return (1-x_frac) * (1-y_frac) * get_val(x_int  , y_int  ) +
             (1-x_frac) *    y_frac  * get_val(x_int  , y_int+1) +
                x_frac  * (1-y_frac) * get_val(x_int+1, y_int  ) +
                x_frac  *    y_frac  * get_val(x_int+1, y_int+1);
    }
    
    // Resamples the image to an X times Y pixel image with new pixel
    // size (x_step, y_step) relative to the current pixel size (1, 1)
    // The top left pixel of the new image is given by (offset_x, offset_y),
    // in terms of the current pixel size (1, 1)
    Image<T> resampleBilinear(const RealType x_step,
                              const RealType y_step,
                              const int X,
                              const int Y,
                              const RealType offset_x,
                              const RealType offset_y,
                              const T zero_value) const {
      Image<T> res(X, Y, zero_value);
      for (int y=0; y<Y; y++)
		for (int x=0; x<X; x++) {
		  const RealType x_coord = offset_x + x * x_step;
		  const RealType y_coord = offset_y + y * y_step;
		  
		  res(x, y) = this->interpolate(x_coord, y_coord, zero_value);
		}
	  
	  return res;
    }
    
    // Same as resample Bilinear, but with periodic distance to the
    // image center (X/2, Y/2). Compare also to the corresponding methods
    // of the CircularImage class.
    Image<T> resampleBilinearPeriodic(const RealType x_step,
                                      const RealType y_step,
                                      const int X,
                                      const int Y,
                                      const RealType offset_x,
                                      const RealType offset_y,
                                      const T zero_value) const {
      const RealType lenX = X * x_step;
      const RealType lenY = Y * y_step;
      
      Image<T> res(X, Y, zero_value);
      for (int y=0; y<Y; y++)
		for (int x=0; x<X; x++) {
		  RealType x_coord = offset_x + x * x_step;
		  RealType y_coord = offset_y + y * y_step;
		  
		  // Move (x_coord, y_coord) as closely as possible to (X/2, Y/2)
		  // with shifts of lenX in the x coordinate and lenY in the y
		  // coordinate
		  const int a = static_cast<int>((X/2 + lenX/2 - x_coord) / lenX);
		  const int b = static_cast<int>((Y/2 + lenY/2 - y_coord) / lenY);
		  
		  x_coord += a * lenX;
		  y_coord += b * lenY;
		  
		  res(x, y) = this->interpolate(x_coord, y_coord, zero_value);
		}
	  
	  return res;
    }
    
    friend void impl_FourierTransform(Image<complex>&, Image<complex>*, bool);
    friend void save(const Image<complex>&, const std::string&, const std::string&, const std::string&);
    friend void save(const Image<RealType>&, const std::string&, const std::string&);
};

// Save a 2D floating point image as a netcdf file
void saveNetcdf(const Image<RealType>& img, const std::string& filename) {
  int nc_error;
  int ncid, x_dimid, y_dimid;
  
  if ((nc_error = nc_create(filename.c_str(), NC_CLOBBER, &ncid)))
    Error("Netcdf error: " + std::string(nc_strerror(nc_error)), __FILE__, __LINE__);
  
  if ((nc_error = nc_def_dim(ncid, "x", img.getY(), &x_dimid)))
    Error("Netcdf error: " + std::string(nc_strerror(nc_error)), __FILE__, __LINE__);
  
  if ((nc_error = nc_def_dim(ncid, "y", img.getX(), &y_dimid)))
    Error("Netcdf error: " + std::string(nc_strerror(nc_error)), __FILE__, __LINE__);
  
  int dimids[2];
  dimids[0] = x_dimid;
  dimids[1] = y_dimid;
  
  int varid;
  if ((nc_error = nc_def_var(ncid, "data", NC_DOUBLE, 2, dimids, &varid)))
    Error("Netcdf error: " + std::string(nc_strerror(nc_error)), __FILE__, __LINE__);
  
  if ((nc_error = nc_enddef(ncid)))
    Error("Netcdf error: " + std::string(nc_strerror(nc_error)), __FILE__, __LINE__);
  
  std::vector<double> tmp(img.size());
  for (int i=0; i<img.size(); i++)
    tmp[i] = img[i];
  
  if ((nc_error = nc_put_var_double(ncid, varid, tmp.data())))
    Error("Netcdf error: " + std::string(nc_strerror(nc_error)), __FILE__, __LINE__);
  
  if ((nc_error = nc_close(ncid)))
    Error("Netcdf error: " + std::string(nc_strerror(nc_error)), __FILE__, __LINE__);
}

// Save as a netcdf file
void save(const Image<complex>& img, const std::string& path,
          const std::string& re_dir="",
          const std::string& im_dir_="") {
  std::string im_dir = im_dir_;
  if (im_dir.empty() && !re_dir.empty())
    im_dir = re_dir;
  
  // Create separate real valued images of the real and imaginary part
  Image<RealType> real_part(img.getX(), img.getY(), 0);
  Image<RealType> imag_part(img.getX(), img.getY(), 0);
  
  for (int i=0; i<img.size(); i++) {
    real_part[i] = img[i].real();
    imag_part[i] = img[i].imag();
  }
  
  // Write the real part
  std::string re_path = path + "_re.nc";
  if (!re_dir.empty()) {
    re_path = re_dir + '/' + re_path;
    std::filesystem::create_directories(re_dir);
  }
  
  saveNetcdf(real_part, re_path);
  
  // Write the imaginary part
  std::string im_path = path + "_im.nc";
  if (!im_dir.empty()) {
    im_path = im_dir + '/' + im_path;
    std::filesystem::create_directories(im_dir);
  }
  
  saveNetcdf(imag_part, im_path);
}

// Save RealType-valued grayscale image as a netcdf file
void save(const Image<RealType>& img, const std::string& path, const std::string& dir="") {
  // Write the real part
  std::string filename = path + ".nc";
  if (!dir.empty()) {
    filename = dir + '/' + path + ".nc";
    std::filesystem::create_directories(dir);
  }
  
  saveNetcdf(img, filename);
}

Image<complex> complex_conjugate(const Image<complex>& img) {
  Image<complex> result(img);
  
  for (int i=0; i<result.size(); i++) 
    result[i] = std::conj(result[i]);
  
  return result;
}

template <typename T>
Image<T>& operator-=(Image<T>& lhs, const Image<T>& rhs) {
  if (lhs.getX() != rhs.getX() || lhs.getY() != rhs.getY())
    Error("Cannot subtract two images of different sizes!", __FILE__, __LINE__);
  
  for (int i=0; i<lhs.size(); i++)
    lhs[i] -= rhs[i];
  
  return lhs;
}

template <typename T>
Image<T>& operator-=(Image<T>& lhs, const T& rhs) {
  for (int i=0; i<lhs.size(); i++)
	lhs[i] -= rhs;
  
  return lhs;
}

template <typename T>
Image<T>& operator+=(Image<T>& lhs, const Image<T>& rhs) {
  if (lhs.getX() != rhs.getX() || lhs.getY() != rhs.getY())
    Error("Cannot add two images of different sizes!", __FILE__, __LINE__);
  
  for (int i=0; i<lhs.size(); i++)
    lhs[i] += rhs[i];
  
  return lhs;
}

template <typename T>
Image<T>& operator*=(Image<T>& lhs, const Image<T>& rhs) {
  if (lhs.getX() != rhs.getX() || lhs.getY() != rhs.getY())
    Error("Cannot perform pointwise multiplication of two images of different sizes!", __FILE__, __LINE__);
  
  for (int i=0; i<lhs.size(); i++)
    lhs[i] *= rhs[i];
  
  return lhs;
}

template <typename T>
Image<T>& operator*=(Image<T>& lhs, const T value) {
  for (int i=0; i<lhs.size(); i++)
    lhs[i] *= value;
  
  return lhs;
}

Image<complex>& operator*=(Image<complex>& lhs, const RealType value) {
  for (int i=0; i<lhs.size(); i++)
    lhs[i] *= value;
  
  return lhs;
}

template <typename T>
Image<T>& operator/=(Image<T>& lhs, const T value) {
  for (int i=0; i<lhs.size(); i++)
    lhs[i] /= value;
  
  return lhs;
}

Image<complex>& operator/=(Image<complex>& lhs, const RealType value) {
  lhs *= 1 / value;
  
  return lhs;
}

template <typename T>
T dotProduct(const Image<T>& lhs, const Image<T>& rhs) {
  if (lhs.getX() != rhs.getX() || lhs.getY() != rhs.getY())
    Error("Incompatible image sizes!", __FILE__, __LINE__);
  
  T result = lhs[0] * rhs[0];
  for (int i=1; i<lhs.size(); i++)
    result += lhs[i] * rhs[i];
  
  return result;
}

RealType norm(const Image<complex>& img) {
  RealType res = 0.;
  for (int i=0; i<img.size(); i++)
    res += img[i].real() * img[i].real() + img[i].imag() * img[i].imag();
  return sqrt(res);
}

RealType norm(const Image<RealType>& img) {
  RealType res = 0.;
  for (int i=0; i<img.size(); i++)
    res += img[i] * img[i];
  return sqrt(res);
}

Image<RealType> pointwise_norm(const Image<complex>& img) {
  Image<RealType> res(img.getX(), img.getY(), static_cast<RealType>(0));
  for (int i=0; i<img.size(); i++)
    res[i] = sqrt(img[i].real() * img[i].real() + img[i].imag() * img[i].imag());
  
  return res;
}

RealType sup_norm(const Image<complex>& img) {
  RealType res_sqr = 0.;
  for (int i=0; i<img.size(); i++) {
    RealType abs_sqr = img[i].real() * img[i].real() + img[i].imag() * img[i].imag();
    if (abs_sqr > res_sqr)
      res_sqr = abs_sqr;
  }
  return sqrt(res_sqr);
}

RealType sup_norm(const Image<RealType>& img) {
  RealType res_sqr = 0.;
  for (int i=0; i<img.size(); i++) {
    RealType sqr = img[i] * img[i];
    if (sqr > res_sqr)
      res_sqr = sqr;
  }
  return sqrt(res_sqr);
}

template <typename T>
T sum(const Image<T>& img) {
  if (img.size() == 0)
    Error("Cannot compute sum of entries of an empty image!", __FILE__, __LINE__);
  
  T res = img[0];
  for (int i=1; i<img.size(); i++)
    res += img[i];
  return res;
}

// Calculates the convolution of a complex valued image with a complex valued kernel
// using constant continuation by a given value.
//
// Note: the kernel width and height must be odd
Image<complex> convolve(const Image<complex>& image, const Image<complex>& kernel, const complex image_ext) {
  const int X = image.getX();
  const int Y = image.getY();
  
  const int pX = kernel.getX()/2;
  const int pY = kernel.getY()/2;
  
  if (kernel.getX() != 2*pX + 1 || kernel.getY() != 2*pY + 1)
    Error("Invalid kernel size in convolve()!", __FILE__, __LINE__);
  
  Image<complex> result(X, Y, {0., 0.});
  
  for (int y=0; y<Y; y++)
    for (int x=0; x<X; x++)
      for (int ky=-pY; ky<=pY; ky++)
        for (int kx=-pX; kx<=pX; kx++) {
          const int img_x = x-kx;
          const int img_y = y-ky;
          
          complex image_val;
          if (img_x < 0 || img_x >= X || img_y < 0 || img_y >= Y)
            image_val = image_ext;
          else
            image_val = image(img_x, img_y);
          
          const complex kernel_val = kernel(kx+pX, ky+pY);
          
          result(x, y) += image_val * kernel_val;
        }
  
  return result;
}

// Calculates the convolution of every row of a complex valued image with a complex valued
// one-dimensional kernel
//
// Note: the kernel size must be odd
void convolve_rows(Image<complex>& image,
                   const std::vector<complex>& kernel,
                   const complex image_ext) {
  const int X = image.getX();
  const int Y = image.getY();
  
  const int pX = static_cast<int>(kernel.size())/2;
  
  if (static_cast<int>(kernel.size()) != 2*pX + 1)
    Error("Invalid kernel size in convolve_rows()!", __FILE__, __LINE__);
  
  std::vector<complex> row(X, {0, 0});
  for (int y=0; y<Y; y++) {
    // Copy the current row
    for (int x=0; x<X; x++)
      row[x] = image(x, y);
    
    // Calculate the convolution with the current row
    for (int x=0; x<X; x++)
      for (int kx=-pX; kx<=pX; kx++) {
        const int img_x = x-kx;
        
        complex image_val;
        if (img_x < 0 || img_x >= X)
          image_val = image_ext;
        else
          image_val = row[img_x];
        
        const complex kernel_val = kernel[kx+pX];
        
        image(x, y) += image_val * kernel_val;
      }
  }
}

// Calculates the convolution of every column of a complex valued image with a complex 
// valued one-dimensional kernel
//
// Note: the kernel size must be odd
void convolve_cols(Image<complex>& image,
                   const std::vector<complex>& kernel,
                   const complex image_ext) {
  const int X = image.getX();
  const int Y = image.getY();
  
  const int pY = static_cast<int>(kernel.size())/2;
  
  if (static_cast<int>(kernel.size()) != 2*pY + 1)
    Error("Invalid kernel size in convolve_rows()!", __FILE__, __LINE__);
  
  std::vector<complex> col(Y, {0, 0});
  for (int x=0; x<X; x++) {
    // Copy the current column
    for (int y=0; y<Y; y++)
      col[y] = image(x, y);
    
    // Calculate the convolution with the current column
    for (int y=0; y<Y; y++)
      for (int ky=-pY; ky<=pY; ky++) {
        const int img_y = y-ky;
        
        complex image_val;
        if (img_y < 0 || img_y >= Y)
          image_val = image_ext;
        else
          image_val = col[img_y];
        
        const complex kernel_val = kernel[ky+pY];
        
        image(x, y) += image_val * kernel_val;
      }
  }
}

// Calculates the Fourier transform of a complex valued image (without normalization factor)
// using FFTW
//
// This function should only be used in code that is not performance critical
// because a new plan is generated every time
void impl_FourierTransform(Image<complex>& source,
                           Image<complex> *dest,
                           bool inverse_fft) {
  if (!(source.X == dest->X && source.Y == dest->Y))
    Error("The input and output images for the FFT must have the same size!", __FILE__, __LINE__);
  
  fftw_plan p;
  
  #pragma omp critical (fftw_function)
  {
    p = fftw_plan_dft_2d(source.Y,
                         source.X,
                         reinterpret_cast<fftw_complex*>(&source.data[0]),
                         reinterpret_cast<fftw_complex*>(&dest->data[0]),
                         (inverse_fft ? FFTW_BACKWARD : FFTW_FORWARD),
                         FFTW_ESTIMATE);
  }
  
  // Use FFTW to calculate the DFT
  fftw_execute(p);
  
  #pragma omp critical (fftw_function)
  {
	fftw_destroy_plan(p);
  }
}

// Forward Fourier transform (unnormalized transform)
void FourierTransform(Image<complex>& source, Image<complex> *dest) {
  impl_FourierTransform(source, dest, false);
}

// Inverse Fourier transform (unnormalized transform)
void InvFourierTransform(Image<complex>& source, Image<complex> *dest) {
  impl_FourierTransform(source, dest, true);
}

// Applies an elliptical bandwidth limit
//
// Note: rx and ry are in fractional coordinates, where a value of 1 corresponds to a
//       radius of img->X/2 resp. img->Y/2.
void applyBandwidthLimit(Image<complex> *img, const RealType rx=1., const RealType ry=1.) {
  const int X = img->getX();
  const int Y = img->getY();
  
  // Transform to Fourier space
  Image<complex> img_fs(*img);
  FourierTransform(*img, &img_fs);
  img_fs.applyFourierShift();
  
  // Apply the bandwidth limit
  const RealType center_x = X/2;
  const RealType center_y = Y/2;
  for (int y=0; y<Y; y++)
    for (int x=0; x<X; x++) {
      const RealType dx = (x - center_x) / (X/2);
      const RealType dy = (y - center_y) / (Y/2);
      
      if (dx*dx / (rx*rx) + dy*dy / (ry*ry) > 1)
        img_fs(x, y) = 0;
    }
  
  // Transform to real space and normalize
  img_fs.applyInverseFourierShift();
  InvFourierTransform(img_fs, img);
  
  *img /= static_cast<RealType>(X * Y);
}

// Applies an elliptical bandwidth limit
void applyBandwidthLimit(Image<RealType> *img, const RealType rx=1., const RealType ry=1.) {
  const int X = img->getX();
  const int Y = img->getY();
  
  Image<complex> img_copy(X, Y, complex(0, 0));
  for (int i=0; i<img->size(); i++)
	img_copy[i] = complex((*img)[i], 0);
  
  applyBandwidthLimit(&img_copy, rx, ry);
  
  for (int i=0; i<img->size(); i++)
    (*img)[i] = img_copy[i].real();
}

// Return an image consisting of every k-th pixel of the input image in x and y direction
template <typename T>
Image<T> subsampleImage(const Image<T>& img, const int k) {
  if (k < 1)
    Error("Invalid image subsampling stepsize!", __FILE__, __LINE__);
  
  if (img.size() == 0)
	return img;
  
  const int X = std::max(std::min(1, img.getX()), img.getX() / k);
  const int Y = std::max(std::min(1, img.getY()), img.getY() / k);
  
  Image<T> res(X, Y, img(0, 0));
  
  for (int y=0; y<Y; y++)
    for (int x=0; x<X; x++)
      res(x, y) = img(k*x, k*y);
  
  return res;
}

Image<RealType> getIndicatorImage(const std::vector<std::array<int, 2>>& pixel_set,
                                  const int X,
                                  const int Y) {
  Image<RealType> res(X, Y, 0);
  
  for (auto& coord: pixel_set)
    res.set(coord, 1);
  
  return res;
}

#endif  // multislice_utility_image_h
