// Container describing a rectangular standard lattice in 2D with a finite number of
// points
#ifndef multislice_utility_lattice_h
#define multislice_utility_lattice_h

#include "utility.h"

#include <vector>
#include <utility>
#include <algorithm>
#include <array>
#include <cmath>

struct Lattice {
  // Number of lattice points in x and y direction
  int X;
  int Y;
  
  // Distance of neighboring lattice points in x and y direction in Angstrom
  RealType dx;
  RealType dy;
  
  // Offset of the top left (0, 0) lattice point in Angstrom
  RealType offset_x;
  RealType offset_y;
  
  // Returns the position of a given lattice point (x, y) in Angstrom
  //
  // Note: the returned position is normalized to lie within [0, X*dx) x [0, Y*dy)
  std::array<RealType, 2> getPosition(int x, int y) const {
    RealType pos_x = x * dx + offset_x;
    RealType pos_y = y * dy + offset_y;
    
    const RealType lenX = X*dx;
    const RealType lenY = Y*dy;
    
    pos_x -= std::floor(pos_x / lenX) * lenX;
    pos_y -= std::floor(pos_y / lenY) * lenY;
  
    return {pos_x, pos_y};
  }
  
  // Returns a vector of all lattice points that are within a distance of dist Angstrom of
  // (pos[0], pos[1]) assuming periodic continuation of the lattice, sorted in ascending
  // order with respect to the distance to pos
  std::vector<std::array<int, 2>> getNearbyPointsPeriodic(std::array<RealType, 2> pos,
                                                          const RealType dist) const {
    const RealType lenX = X * dx;
    const RealType lenY = Y * dy;
    
    const RealType dist_sqr = dist * dist;
    
    // Ensure that (pos_x, pos_y) is within [0, lenX) x [0, lenY) using the periodicity
    pos[0] -= std::floor(pos[0] / lenX) * lenX;
    pos[1] -= std::floor(pos[1] / lenY) * lenY;
    
    // Iterate over all lattice points to find those that are close to (pos_x, pos_y)
    std::vector<std::pair<std::array<int, 2>, RealType>> indices_unsorted;
    indices_unsorted.reserve(X * Y);
    
    for (int y=0; y<Y; y++)
      for (int x=0; x<X; x++) {
        // Position of the lattice point (x, y) in Angstrom
        std::array<RealType, 2> lattice_pos = getPosition(x, y);
        
        // Squared distance to (pos[0], pos[1]) with periodic boundary conditions
        const RealType dx_sqr = periodicSquaredDist(pos[0], lattice_pos[0], lenX);
        const RealType dy_sqr = periodicSquaredDist(pos[1], lattice_pos[1], lenY);
        
        if (dx_sqr + dy_sqr < dist_sqr)
          indices_unsorted.push_back(std::make_pair(std::array<int, 2>{x, y}, dx_sqr + dy_sqr));
      }
    
    // Sort indices by distance to pos
    std::sort(indices_unsorted.begin(), indices_unsorted.end(),
              [](auto a, auto b) {
                return a.second < b.second;
              }
             );
    
    // Copy the sorted indices to an appropriate data structure
    std::vector<std::array<int, 2>> result(indices_unsorted.size());
    for (unsigned int i=0; i<result.size(); i++)
      result[i] = indices_unsorted[i].first;
    
    return result;
  }
};

#endif  // multislice_utility_lattice_h
