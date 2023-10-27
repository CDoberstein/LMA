// Container class to store 2D images with circular symmetry
#ifndef multislice_utility_circularimage_h
#define multislice_utility_circularimage_h

#include "Image.h"
#include "utility.h"

#include <vector>
#include <algorithm>

template <typename T>
class CircularImage {
  private:
    // Equidistantly spaced image values sorted ascendingly by distance from the center
    std::vector<T> line;
    
    // Size of one "pixel" in the line vector
    RealType pixel_size;
  
  public:
    CircularImage(const std::vector<T>& values = std::vector<T>(),
                  const RealType pixel_size = 0.)
      : line(values), pixel_size(pixel_size) { }
    
    bool empty() const {
      return line.empty();
    }
    
    // Returns the image value at an arbitrary distance from the center using linear
    // interpolation
    T operator[](const RealType dist) const {
      const int N = static_cast<int>(line.size());
      
      const RealType px_dist = dist / pixel_size;
      
      int dist_int = static_cast<int>(px_dist);
      RealType dist_frac = px_dist - dist_int;
      
      if (dist_int >= N) {
        dist_int = N - 1;
        dist_frac = 0;
      }
      
      T v1 = line[dist_int];
      T v2 = line[(dist_int+1 < N) ? (dist_int+1) : dist_int];
      
      v1 *= (1 - dist_frac);
      v2 *= dist_frac;
      
      return v1 + v2;
    }
    
    // Returns the maximum distance for which *this stores image values
    RealType getMaxDist() const {
      return (line.size()-1) * pixel_size;
    }
    
    // Convert to an ordinary 2D image
    Image<T> get2DImageNonSquare(const RealType sampling_pixel_size_x,
                                 const RealType sampling_pixel_size_y,
                                 const int X,
                                 const int Y,
                                 const RealType pos_x,
                                 const RealType pos_y,
                                 const T zero_value,
                                 RealType radius = -1) const {
      const RealType maxDist = getMaxDist();
      
      if (radius == -1)
        radius = maxDist;
      
      Image<T> res(X, Y, zero_value);
      for (int y=0; y<Y; y++)
        for (int x=0; x<X; x++) {
          RealType dx_sqr = (x * sampling_pixel_size_x - pos_x) * (x * sampling_pixel_size_x - pos_x);
          RealType dy_sqr = (y * sampling_pixel_size_y - pos_y) * (y * sampling_pixel_size_y - pos_y);
          
          const RealType dist = sqrt(dx_sqr + dy_sqr);
          
          if (dist > radius || dist > maxDist)
            continue;
          
          res(x, y) = this->operator[](dist);
        }
      
      return res;
    }
    
    Image<T> get2DImageNonSquare(const RealType sampling_pixel_size_x,
                                 const RealType sampling_pixel_size_y,
                                 const int X,
                                 const int Y,
                                 const T zero_value,
                                 RealType radius = -1) const {
      return get2DImageNonSquare(sampling_pixel_size_x, sampling_pixel_size_y, X, Y,
                                 sampling_pixel_size_x * (X/2), sampling_pixel_size_y * (Y/2),
                                 zero_value, radius);
    }
    
    Image<T> get2DImageNonSquare(const RealType sampling_pixel_size_x,
                                 const RealType sampling_pixel_size_y,
                                 const T zero_value,
                                 RealType radius = -1) const {
      if (radius == -1)
        radius = getMaxDist();
      
      const int X = 2 * static_cast<int>(radius / sampling_pixel_size_x);
      const int Y = 2 * static_cast<int>(radius / sampling_pixel_size_y);
      
      return get2DImageNonSquare(sampling_pixel_size_x, sampling_pixel_size_y, X, Y, zero_value, radius);
    }
    
    Image<T> get2DImage(const RealType sampling_pixel_size,
                        const T zero_value,
                        const RealType radius = -1) const {
      return get2DImageNonSquare(sampling_pixel_size, sampling_pixel_size, zero_value, radius);
    }
    
    // Same as the above get2DImage(NonSquare) methods, but the distance to
    // the center (pos_x, pos_y) is calculated with periodic continuation.
    // Compare also to the corresponding methods of the Image class
    Image<T> get2DImageNonSquarePeriodic(const RealType sampling_pixel_size_x,
                                         const RealType sampling_pixel_size_y,
                                         const int X,
                                         const int Y,
                                         const RealType pos_x,
                                         const RealType pos_y,
                                         const T zero_value,
                                         RealType radius = -1) const {
      const RealType maxDist = getMaxDist();
      
      if (radius == -1)
        radius = maxDist;
      
      // (pos_x, pos_y) is expected to be given in terms of the sampling
      // pixel size. Convert to pixel coordinates:
      const RealType center_x = pos_x / sampling_pixel_size_x;
      const RealType center_y = pos_y / sampling_pixel_size_y;
      
      Image<T> res(X, Y, zero_value);
      for (int y=0; y<Y; y++)
        for (int x=0; x<X; x++) {
          // Note: here, the distance is calculated with periodic continuation at the image borders
          const RealType dx_sqr = periodicSquaredDist(x, center_x, X) * sampling_pixel_size_x * sampling_pixel_size_x;
          const RealType dy_sqr = periodicSquaredDist(y, center_y, Y) * sampling_pixel_size_y * sampling_pixel_size_y;
          
          const RealType dist = sqrt(dx_sqr + dy_sqr);
          
          if (dist > radius || dist > maxDist)
            continue;
          
          res(x, y) = this->operator[](dist);
        }
      
      return res;
    }
    
    std::vector<T> getData() const { return line; }
    RealType getPixelSize() const { return pixel_size; }
};

#endif  // multislice_utility_circularimage_h
