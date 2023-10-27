// Container class to store 3D images
#ifndef multislice_utility_image3d_h
#define multislice_utility_image3d_h

#include "Image.h"
#include "complex.h"
#include "Error.h"

#include <vector>
#include <filesystem>
#include <algorithm>

template <typename T> class Image3D;

void save(const Image3D<RealType>&, const std::string&, const std::string&);
Image<RealType> getIntegrated2DImage(const Image3D<RealType>&, const int, const int);
RealType sup_dist(const Image3D<RealType>&, const Image3D<RealType>&);
RealType euc_dist(const Image3D<RealType>&, const Image3D<RealType>&);
RealType sup_norm(const Image3D<RealType>&);
RealType euc_norm(const Image3D<RealType>&);

template <typename T>
class Image3D {
  private:
    // All data points in a single vector
    std::vector<T> data;
    
    // Length of the first dimension
    int X;
    
    // Product of the lengths of the first and the second dimension
    int XY;
  public:
    Image3D() : X(0), XY(0) {};
    Image3D(const int X, const int Y, const int Z) : data(X*Y*Z, static_cast<T>(0)), X(X), XY(X*Y) {}
    
    T& operator()(const int x, const int y, const int z) {
      return data[x + y*X + z*XY];
    }
    const T& operator()(const int x, const int y, const int z) const {
      return data[x + y*X + z*XY];
    }
    
    T operator[](const int i) const { return data[i]; }
    T& operator[](const int i) { return data[i]; }
    
    int getX() const { return X; }
    int getY() const { return XY / X; }
    int getZ() const { return static_cast<int>(data.size()) / XY; }
    
    bool empty() const {
      return data.empty();
    }
    
    int size() const { return static_cast<int>(data.size()); }
    
    friend void save(const Image3D<RealType>&, const std::string&, const std::string&);
    friend Image<RealType> getIntegrated2DImage(const Image3D<RealType>&, const int, const int);
    friend RealType sup_dist(const Image3D<RealType>&, const Image3D<RealType>&);
    friend RealType euc_dist(const Image3D<RealType>&, const Image3D<RealType>&);
    friend RealType sup_norm(const Image3D<RealType>&);
    friend RealType euc_norm(const Image3D<RealType>&);
};

// Saves the 3D image as individual 2D slices along the Z direction
void save(const Image3D<RealType>& img,
          const std::string& filename,
          const std::string& directory) {
  const int X = img.X;
  const int Y = img.XY / img.X;
  const int Z = static_cast<int>(img.data.size()) / img.XY;
  
  std::filesystem::create_directories(directory);
  for (int z=0; z<Z; z++) {
    Image<RealType> slice(X, Y, 0);
    for (int i=0; i<img.XY; i++)
      slice[i] = img.data[i + z*img.XY];
    
    saveNetcdf(slice, directory + '/' + filename + '_' + std::to_string(z) + ".nc");
  }
}

// Integrates a 3D image along the z axis
Image<RealType> getIntegrated2DImage(const Image3D<RealType>& img,
                              const int min_z,
                              const int max_z) {
  const int X = img.X;
  const int Y = img.XY / img.X;
  const int Z = static_cast<int>(img.data.size()) / img.XY;
  
  Image<RealType> result(X, Y, static_cast<RealType>(0));
  
  if (min_z > max_z || min_z >= Z)
    return result;
  
  for (int z=min_z; z<std::min(max_z, Z); z++)
    for (int y=0; y<Y; y++)
      for (int x=0; x<X; x++)
        result(x, y) += img(x, y, z);
  
  return result;
}

// Calculates the supremum norm of the difference of two 3D arrays
RealType sup_dist(const Image3D<RealType>& a, const Image3D<RealType>& b) {
  if (a.X != b.X || a.XY != b.XY || a.data.size() != b.data.size())
    Error("Incompatible 3D image sizes!", __FILE__, __LINE__);
  
  if (a.data.empty())
    return 0;
  
  RealType result_sqr = 0;
  for (unsigned int i=0; i<a.data.size(); i++) {
    const RealType sqr_diff = (a.data[i] - b.data[i]) * (a.data[i] - b.data[i]);
    
    if (sqr_diff > result_sqr)
      result_sqr = sqr_diff;
  }
  
  return sqrt(result_sqr);
}

// Calculates the euclidean norm of the difference of two 3D arrays
RealType euc_dist(const Image3D<RealType>& a, const Image3D<RealType>& b) {
  if (a.X != b.X || a.XY != b.XY || a.data.size() != b.data.size())
    Error("Incompatible 3D image sizes!", __FILE__, __LINE__);
  
  if (a.data.empty())
    return 0;
  
  RealType result_sqr = 0;
  for (unsigned int i=0; i<a.data.size(); i++)
    result_sqr += (a.data[i] - b.data[i]) * (a.data[i] - b.data[i]);
  
  return sqrt(result_sqr);
}

// Calculates the supremum norm of a 3D array
RealType sup_norm(const Image3D<RealType>& a) {
  RealType result_sqr = 0;
  for (unsigned int i=0; i<a.data.size(); i++) {
    const RealType sqr = a.data[i] * a.data[i];
    
    if (sqr > result_sqr)
      result_sqr = sqr;
  }
  
  return sqrt(result_sqr);
}

// Calculates the euclidean norm of a 3D array
RealType euc_norm(const Image3D<RealType>& a) {
  RealType result_sqr = 0;
  for (unsigned int i=0; i<a.data.size(); i++)
    result_sqr += a.data[i] * a.data[i];
  
  return sqrt(result_sqr);
}

template <typename T>
Image3D<T>& operator/=(Image3D<T>& lhs, const T value) {
  for (int i=0; i<lhs.size(); i++)
    lhs[i] /= value;
  
  return lhs;
}

#endif  // multislice_utility_image3d_h
