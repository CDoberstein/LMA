// The function chooseLMAStrategy() attemps find a good strategy that
// minimizes the number of calls to the Multislice algorithm
// in Compute3DSTEM() while at the same time respecting the memory limit
// set by the memory_limit_MB parameter
#ifndef multislice_lmastrategy_h
#define multislice_lmastrategy_h

#include "LMAStrategy_aux.h"

#include "OutputWindow.h"

#include "../Utility/Image.h"
#include "../Utility/Lattice.h"

#include <vector>
#include <utility>
#include <cmath>
#include <array>
#include <algorithm>
#include <set>
#include <string>
#include <iomanip>

struct LMAStrategy {
  // Name of this strategy (e.g. "trivial", "row", "greedy" or "box")
  std::string name;
  
  // Total number of multislice computations required for the
  // calculation of the resulting STEM image using this strategy
  int num_multislice_computations;
  
  // Computation domains for each step: each domain contains a set of
  // probe positions (discrete coordinates with respect to the output
  // window) for which the corresponding pixel value in the final STEM
  // image is to be computed in this step.
  //
  // Note: these domains form a partition of the output window.
  typedef std::vector<std::array<int, 2>> CompDomain;
  
  std::vector<CompDomain> computation_domains;
  
  LMAStrategy() : num_multislice_computations(0) {}
  
  LMAStrategy(const std::array<int, 2>& first_point)
    : num_multislice_computations(0) {
    computation_domains.resize(1, CompDomain(1, first_point));
  }
  
  void calculateTotalCost(const Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                          const Lattice& input_wave_lattice,
                          const Lattice& probe_lattice,
                          const OutputWindow& output_window,
                          const Param& p) {
    // Go through the computation domains just like the Lattice
    // Multislice algorithm would, without actually performing any
    // Multislice computations
    auto cmp = [&](const std::array<int, 2>& a, const std::array<int, 2>& b) {
      return (a[0] + a[1] * input_wave_lattice.X < b[0] + b[1] * input_wave_lattice.X);
    };
    
    num_multislice_computations = 0;
    
    std::vector<std::array<int, 2>> prev_iw_pos, cur_iw_pos;
    for (const auto& cd: computation_domains) {
      cur_iw_pos = getRequiredInputWavePositions(cd,
                                                 CoefficientIndices,
                                                 input_wave_lattice,
                                                 probe_lattice,
                                                 output_window.start_x,
                                                 output_window.start_y,
                                                 p);
      
      std::vector<std::array<int, 2>> to_compute;
      std::set_difference(cur_iw_pos.begin(), cur_iw_pos.end(),
                          prev_iw_pos.begin(), prev_iw_pos.end(),
                          std::inserter(to_compute, to_compute.begin()),
                          cmp);
      
      num_multislice_computations += to_compute.size();
      
      prev_iw_pos = cur_iw_pos;
    }
  }
  
  void saveAsImage(const OutputWindow& output_window,
                   const std::string& filename,
                   const std::string& dir) const {
    Image<RealType> img(output_window.X, output_window.Y, 0);
    
    for (int i=0; i<static_cast<int>(computation_domains.size()); i++)
      for (const std::array<int, 2>& p: computation_domains[i])
        img(p) = i+1;
    
    save(img, filename, dir);
  }
  
  // Returns an image where all pixels that are contained in the first
  // k computation domains are set to 1 and all others are set to 0
  Image<RealType> getProgressImage(int k, const OutputWindow& output_window) const {
    Image<RealType> progress_image(output_window.X, output_window.Y, 0);
    
    k = std::clamp(k, 0, static_cast<int>(computation_domains.size()));
    for (int i=0; i<k; i++)
      for (const std::array<int, 2>& p: computation_domains[i])
        progress_image(p) = 1;
    
    return progress_image;
  }
};

// Greedy algorithm that attempts to compute a good partition of the
// output window into subsets such that
//
//  (1) the number of Multislice results required for the computation of
//      all probe positions within a given subset must not exceed
//      result_storage_limit
//
//  (2) the number of times that the Multislice algorithm needs to be
//      executed is minimized, taking into account that results may be
//      reused when passing from one subset to the next.
LMAStrategy greedyLMAStrategy(const OutputWindow& output_window,
                              const Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                              const Lattice& input_wave_lattice,
                              const Lattice& probe_lattice,
                              const int result_storage_limit,
                              const int lma_strategy_computation_level,
                              const bool verboseLMASearch,
                              const Param& param) {
  // The greedy algorithm may be performed multiple times depending on
  // the value of the lma_strategy_computation_level parameter. The only
  // difference between the different runs of the algorithm are
  // different choices for the first pixel to be computed. These pixels
  // are chosen to lie on a regular grid with
  // num_start_pixels x num_start_pixels points, which is placed on top
  // of the output window.
  //
  // The value of num_start_pixels is 2^k - 2^(k-1) + 1, where k is
  // lma_strategy_computation_level. This formula ensures that for
  // k'>k, the regular grid corresponding to k' contains all points of
  // the grid corresponding to k. If k is zero, then the greedy
  // algorithm is performed only once with the pixel in the top left
  // corner of the output window as the starting point.
  int num_start_pixels;
  
  if (lma_strategy_computation_level <= 0)
    num_start_pixels = 1;
  else if (lma_strategy_computation_level < 30)
    num_start_pixels = (1 << lma_strategy_computation_level) - (1 << (lma_strategy_computation_level-1)) + 1;
  else
    num_start_pixels = (1 << 30);
  
  const int nX = std::min(output_window.X, num_start_pixels);
  const int nY = std::min(output_window.Y, num_start_pixels);
  
  LMAStrategy best_strategy;
  
  // Perform the greedy algorithm nX * nY times
  for (int y=0; y<nY; y++)
    for (int x=0; x<nX; x++) {
      if (!verboseLMASearch)
        std::cerr << "\r\t\t" << (100.*(x + y * nX)) / (nX * nY) << "%    \r";
      
      // true if pixel is not yet included in the partition, false otherwise
      Image<bool> todo(output_window.X, output_window.Y, true);
      
      // true if the Multislice solution for this point in the input
      // wave lattice has been computed / is available, false otherwise
      Image<bool> zero_cost(input_wave_lattice.X, input_wave_lattice.Y, false);
      
      // Compute first pixel
      const std::array<int, 2> first_point = {(nX == 1 ? 0 : x * (output_window.X - 1) / (nX - 1)),
                                              (nY == 1 ? 0 : y * (output_window.Y - 1) / (nY - 1))};
      
      // Initialize the current strategy by adding the first point in the
      // output window
      LMAStrategy current_strategy(first_point);
      todo.set(first_point, false);
      int point_count = 1;
      
      int current_domain_num = 0;
      int current_domain_cost = calculateCost(first_point, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
      setZeroCost(first_point, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
      
      bool current_domain_cleanup_done = true;
      
      // nearby_points contains all neighboring points of the points in
      // current_strategy.computation_domains[current_domain_num] that
      // are (1) contained in the output window and (2) not yet included
      // in any computation domain of the partition
      auto cmp = [&](const std::array<int, 2>& a, const std::array<int, 2>& b) {
        return (a[0] + a[1] * probe_lattice.X < b[0] + b[1] * probe_lattice.X);
      };
      
      std::set<std::array<int, 2>, decltype(cmp)> nearby_points(cmp);
      
      auto addNearbyPoints = [&](const std::array<int, 2>& p) {
        // Adds all points that are immediately adjacent to p (including
        // diagonals) to nearby_points, given that these points lie
        // within the output window and have not been added to the
        // partition yet.
        for (int yy=-1; yy<=1; yy++)
          for (int xx=-1; xx<=1; xx++) {
            // Calculate neighboring positions in the output window,
            // respecting the periodicity of the probe lattice (which is
            // due to the periodicity of the specimen)
            const int nx = (p[0] + xx + output_window.start_x) % probe_lattice.X - output_window.start_x;
            const int ny = (p[1] + yy + output_window.start_y) % probe_lattice.Y - output_window.start_y;
            
            if (nx >= 0 && nx < output_window.X &&
                ny >= 0 && ny < output_window.Y)
              if (todo.get({nx, ny}))
                nearby_points.insert({nx, ny});
          }
      };
      auto addClosePoint = [&](const LMAStrategy::CompDomain& cur_domain) {
        // Finds a point close to the points in the current computation
        // domain and adds it to nearby_points.
        
        // First compute a bounding box of the points in cur_domain
        int x_min = output_window.X;
        int x_max = -1;
        
        int y_min = output_window.Y;
        int y_max = -1;
        
        for (const auto& p: cur_domain) {
          x_min = std::min(x_min, p[0]);
          x_max = std::max(x_max, p[0]+1);
          
          y_min = std::min(y_min, p[1]);
          y_max = std::max(y_max, p[1]+1);
        }
        
        // Check if the bounding box contains a point that has not yet
        // been added to the partition
        for (int yy=y_min; yy<y_max; yy++)
          for (int xx=x_min; xx<x_max; xx++)
            if (todo.get({xx, yy})) {
              nearby_points.insert({xx, yy});
              return;
            }
        
        // Successively increase the size of the bounding box until it
        // contains a point that has not yet been added to the partition
        while (x_min > 0 || x_max < output_window.X ||
               y_min > 0 || y_max < output_window.Y) {
          x_min = std::max(x_min-1, 0);
          x_max = std::min(x_max+1, output_window.X);
          
          y_min = std::max(y_min-1, 0);
          y_max = std::min(y_max+1, output_window.Y);
          
          for (int yy=y_min; yy<y_max; yy++) {
            if (todo.get({x_min, yy})) { nearby_points.insert({x_min, yy}); return; }
            if (todo.get({x_max-1, yy})) { nearby_points.insert({x_max-1, yy}); return; }
          }
          
          for (int xx=x_min; xx<x_max; xx++) {
            if (todo.get({xx, y_min})) { nearby_points.insert({xx, y_min}); return; }
            if (todo.get({xx, y_max-1})) { nearby_points.insert({xx, y_max-1}); return; }
          }
        }
      };
      
      addNearbyPoints(first_point);
      
      // Loop to repeat adding computation domains
      while (point_count < output_window.X * output_window.Y) {
        if (verboseLMASearch)
          std::cerr << "\r\t\t" << x + y*nX << "/" << nX*nY << ": "
                    << (100. * point_count) / (output_window.X * output_window.Y) << "%     \r";
      
        // Loop to add as many points as possible to the current computation domain
        while (current_domain_cost < result_storage_limit) {
          // Make sure there are any points to add in nearby_points
          if (nearby_points.empty()) {
            if (point_count == output_window.X * output_window.Y)
              break;
            addClosePoint(current_strategy.computation_domains[current_domain_num]);
          }
          
          // Find the point in nearby_points that can be added to the
          // current computation domain with the lowest cost
          std::array<int, 2> optimal_point;
          int min_cost = input_wave_lattice.X * input_wave_lattice.Y + 1;
          for (auto it = nearby_points.cbegin(); it != nearby_points.cend(); it++) {
            const int cur_cost = calculateCost(*it, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
            if (cur_cost < min_cost) {
              min_cost = cur_cost;
              optimal_point = *it;
            }
          }
          
          // Add the optimal point if possible
          if (min_cost + current_domain_cost <= result_storage_limit) {
            current_strategy.computation_domains[current_domain_num].push_back(optimal_point);
            nearby_points.erase(optimal_point);
            todo.set(optimal_point, false);
            ++point_count;
            
            current_domain_cost += min_cost;
            setZeroCost(optimal_point, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
            
            addNearbyPoints(optimal_point);
          } else {
            // Impossible to add another point - continue with new
            // computation domain
            break;
          }
          
          // Complete the "cleanup" of calculations of the previous
          // computation domain once the number of points in the current
          // computation domain exceeds 20% of the number of points in the
          // previous computation domain
          if (!current_domain_cleanup_done) {
            if (current_strategy.computation_domains[current_domain_num].size() >
                current_strategy.computation_domains[current_domain_num-1].size() * 0.2) {
              // Reset current_domain_cost, zero_cost array and the
              // nearby_points set
              current_domain_cost = 0;
              zero_cost = Image<bool>(input_wave_lattice.X, input_wave_lattice.Y, false);
              nearby_points.clear();
              
              // Update zero_cost array and current_domain_cost to reflect
              // the points added to the current computation domain so far.
              // Additionally, update nearby points to only include points
              // near those within the current computation domain.
              for (const auto& p: current_strategy.computation_domains[current_domain_num]) {
                current_domain_cost += calculateCost(p, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
                setZeroCost(p, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
                addNearbyPoints(p);
              }
              
              current_domain_cleanup_done = true;
            }
          }
        }
        
        // Stop if the partition is complete
        if (point_count == output_window.X * output_window.Y)
          break;
        
        // Partial "cleanup" for the next computation domain
        // The points in the next computation domain are first found by
        // pretending that all multislice results from this domain are
        // still available
        current_domain_cost = 0;
        
        current_strategy.computation_domains.push_back(LMAStrategy::CompDomain());
        ++current_domain_num;
        
        current_domain_cleanup_done = false;
        
        // Make sure that nearby points is not empty in order to find the
        // first point of the next computation domain in the next iteration
        if (nearby_points.empty())
          addClosePoint(current_strategy.computation_domains[current_domain_num-1]);
      }
      
      // Calculate total cost of current strategy
      current_strategy.calculateTotalCost(CoefficientIndices, input_wave_lattice, probe_lattice, output_window, param);
      
      if (verboseLMASearch)
        std::cerr << "\r\t\t" << x + y*nX << "/" << nX*nY << ": 100% (cost: "
                  << current_strategy.num_multislice_computations << ")";
      
      // Update the best known strategy if the last strategy is
      // better than the previously known best strategy.
      if (x == 0 && y == 0) {
        best_strategy = current_strategy;
      } else {
        if (current_strategy.num_multislice_computations < best_strategy.num_multislice_computations) {
          if (verboseLMASearch)
            std::cerr << " <- new best strategy";
          
          best_strategy = current_strategy;
        }
      }
      
      if (verboseLMASearch)
        std::cerr << std::endl;
    }
    
  if (verboseLMASearch)
    std::cerr << "\t\tNumber of computation domains of the best strategy " << best_strategy.computation_domains.size() << std::endl;
  else
    std::cerr << "\r\t\t100%         " << std::endl;
  
  best_strategy.name = "Greedy";
  return best_strategy;
}

// Returns the trivial strategy to compute all required input waves for
// all probe positions in the output window at once
LMAStrategy trivialLMAStrategy(const OutputWindow& output_window,
                               const Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                               const Lattice& input_wave_lattice,
                               const Lattice& probe_lattice,
                               const Param& p) {
  LMAStrategy trivial_strategy;
  
  trivial_strategy.computation_domains.resize(1, LMAStrategy::CompDomain(output_window.X * output_window.Y));
  for (int y=0; y<output_window.Y; y++)
    for (int x=0; x<output_window.X; x++)
      trivial_strategy.computation_domains[0][x + y*output_window.X] = {x, y};
  
  trivial_strategy.calculateTotalCost(CoefficientIndices, input_wave_lattice, probe_lattice, output_window, p);
  
  trivial_strategy.name = "Trivial";
  return trivial_strategy;
}

// Returns an LMA strategy based on computing rows of the output window
// in each step
LMAStrategy rowLMAStrategy(const OutputWindow& output_window,
                           const Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                           const Lattice& input_wave_lattice,
                           const Lattice& probe_lattice,
                           const int result_storage_limit,
                           const bool verboseLMASearch,
                           const Param& param) {
  LMAStrategy row_strategy;
  row_strategy.computation_domains.push_back(LMAStrategy::CompDomain());
  
  // true if the Multislice solution for this point in the input
  // wave lattice has been computed / is available, false otherwise
  Image<bool> zero_cost(input_wave_lattice.X, input_wave_lattice.Y, false);
  
  int current_domain_num = 0;
  int current_domain_cost = 0;
  
  for (int y=0; y<output_window.Y; y++)
    for (int x=0; x<output_window.X; x++) {
      int xy_cost = calculateCost({x, y}, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
      
      if (current_domain_cost + xy_cost > result_storage_limit) {
        // Create new computation domain
        row_strategy.computation_domains.push_back(LMAStrategy::CompDomain());
        ++current_domain_num;
        current_domain_cost = 0;
        zero_cost = Image<bool>(input_wave_lattice.X, input_wave_lattice.Y, false);
        
        // Recompute xy_cost to account for the change in the zero_cost array
        xy_cost = calculateCost({x, y}, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
      }
      
      // Add {x, y} to the current computation domain
      row_strategy.computation_domains[current_domain_num].push_back({x, y});
      current_domain_cost += xy_cost;
      setZeroCost({x, y}, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
    }
  
  if (verboseLMASearch)
    std::cerr << "\t\tNumber of computation domains: " << row_strategy.computation_domains.size() << std::endl;
  
  row_strategy.calculateTotalCost(CoefficientIndices, input_wave_lattice, probe_lattice, output_window, param);
  
  row_strategy.name = "Row";
  return row_strategy;
}

// Returns an LMA strategy based on computing rectangular subsets of the
// output window in each step
LMAStrategy rectLMAStrategy(const OutputWindow& output_window,
                            const Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                            const Lattice& input_wave_lattice,
                            const Lattice& probe_lattice,
                            const int result_storage_limit,
                            const bool verboseLMASearch,
                            const Param& param) {
  LMAStrategy best_strategy;
  
  // Iterate over all possible rectangle widths and find the corresponding
  // optimal height that maximizes the size of the rectangle while not exceeding
  // result_storage_limit
  for (int width=1; width<=output_window.X; width++) {
    std::vector<std::array<int, 2>> rectangle;
    
    // Successively increase the height of the current rectangle until
    // the cost exceeds result_storage_limit
    Image<bool> zero_cost(input_wave_lattice.X, input_wave_lattice.Y, false);
    int rect_cost = 0;
    int prev_rect_cost = input_wave_lattice.X * input_wave_lattice.Y + 1;
    
    int y;
    for (y=0; y<output_window.Y; y++) {
      // Add one line of points to the rectangle
      for (int x=0; x<width; x++) {
        rectangle.push_back({x, y});
        rect_cost += calculateCost({x, y}, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
        setZeroCost({x, y}, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
      }
      
      // y is the maximum height for a rectangle with the current width
      // if rect_cost exceeds result_storage_limit
      if (rect_cost > result_storage_limit)
        break;
      
      // Update the previous rectangle cost for the next iteration
      prev_rect_cost = rect_cost;
    }
    
    // At this point, the maximum height of a rectangle with the current
    // width is given by y
    int rect_width = width;
    int rect_height = y;
    
    if (rect_height == 0)
      break;
    
    const RealType pixel_price = prev_rect_cost / static_cast<RealType>(rect_width * rect_height);
    
    if (verboseLMASearch)
      std::cerr << "\t\t" << std::setw(3) << rect_width << "/" << output_window.X << ": "
                  << std::setw(6) << std::setprecision(3) << pixel_price
                  << " (rectangle MS calls / pixel), height = " << std::setw(3) << rect_height;
    else
      std::cerr << "\r\t\tRectangle size: " << rect_width << " x " << rect_height << "    \r";
    
    // Rotate the rectangle if necessary so that the shortest rectangle
    // dimension corresponds to the longest output window dimension
    if ((output_window.X > output_window.Y && rect_width > rect_height) ||
        (output_window.Y > output_window.X && rect_height > rect_width)) {
      std::swap(rect_width, rect_height);
    }
    
    // Fill the output window with as many rectangles of size
    // rect_width x rect_height as possible
    LMAStrategy rect_strategy;
    Image<bool> todo(output_window.X, output_window.Y, true);
    int remaining_pixels = output_window.X * output_window.Y;
    
    auto add_rectangle = [&](const int px, const int py) {
      // Auxiliary lambda function to add a new rectangular computation
      // domain of size rect_width x rect_height with top left pixel
      // at (px, py) to rect_strategy
      LMAStrategy::CompDomain rectangle(rect_width * rect_height);
      
      for (int yy=0; yy<rect_height; yy++)
        for (int xx=0; xx<rect_width; xx++) {
          rectangle[xx + yy*rect_width] = {px + xx, py + yy};
          todo.set({px + xx, py + yy}, false);
        }
      
      rect_strategy.computation_domains.push_back(rectangle);
      
      remaining_pixels -= rect_width * rect_height;
    };
    
    const int nX = output_window.X / rect_width;
    const int nY = output_window.Y / rect_height;
    if (rect_width < rect_height) {
      // Fill rows first, then top to bottom.
      //
      // Example of order of rectangles:
      //    1   2   3   4   5   6   7
      //   14  13  12  11  10   9   8
      //   15  16  17  18  ...
      for (int y=0; y<nY; y++)
        for (int x=0; x<nX; x++)
          add_rectangle((y%2 == 0 ? x : nX-1-x) * rect_width, y * rect_height);
    } else {
      // Fill columns first, then left to right.
      //
      // Example of order of rectangles:
      //    1   8   9  16 ...
      //    2   7  10  15
      //    3   6  11  14
      //    4   5  12  13
      for (int x=0; x<nX; x++)
        for (int y=0; y<nY; y++)
          add_rectangle(x * rect_width, (x%2 == 0 ? y : nY-1-y) * rect_height);
    }
    
    // Fill the remaining part of the output window with as many rectangles
    // of size rect_height x rect_width as possible
    const int x_remaining = output_window.X - nX * rect_width;
    const int y_remaining = output_window.Y - nY * rect_height;
    
    std::swap(rect_height, rect_width);
    
    const int nX_rotated = x_remaining / rect_width;
    const int nY_rotated = y_remaining / rect_height;
    
    if (rect_width < rect_height) {
      // Fill columns first
      for (int x=0; x<nX_rotated; x++)
        for (int y=0; y<nY_rotated; y++)
          add_rectangle(((nY-1)%2 == 0 ? nX_rotated-1-x : x) * rect_width,
                        (x%2 == 0 ? y : nY_rotated-1-y) * rect_height + nY * rect_width);
    } else {
      // Fill rows first
      for (int y=0; y<nY_rotated; y++)
        for (int x=0; x<nX_rotated; x++)
          add_rectangle((y%2 == 0 ? x : nX_rotated-1-x) * rect_width + nX * rect_height,
                        ((nX-1)%2 == 0 ? nY_rotated-1-y : y) * rect_height);
    }
    
    // Fill the rest with differently shaped computation domains
    while(remaining_pixels > 0) {
      // Find a pixel that is not part of any computation domain yet
      //
      // Search from top to bottom, right to left, but make sure, that
      // the x coordinate is maximal (that's what the second pair of
      // loops is for).
      int x=0, y=0;
      
      for (y=0; y<output_window.Y; y++) {
        for (x=output_window.X-1; x>=0; x--)
          if (todo.get({x, y}))
            break;
        if (x>=0)
          break;
      }
      
      for (int y2=0; y2<output_window.Y; y2++)
        for (int x2=x+1; x2<output_window.X; x2++)
          if (todo.get({x2, y2})) {
            x = x2;
            y = y2;
          }
      
      // In each step, augment the square around (x, y) by 1 pixel to the
      // left and bottom. If possible, include all pixels within this
      // square that are not yet included in any other computation domain
      // in the current domain. If this is not possible, continue with a
      // new computation domain.
      LMAStrategy::CompDomain current_domain;
      zero_cost = Image<bool>(input_wave_lattice.X, input_wave_lattice.Y, false);
      
      current_domain.push_back({x, y});
      todo.set({x, y}, false);
      --remaining_pixels;
      
      int current_cost = calculateCost({x, y}, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
      setZeroCost({x, y}, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
      
      int square_len = 1;
      while (true) {
        // Collect all points that may be added to current_domain
        std::vector<std::array<int, 2>> new_points;
        new_points.reserve(2*square_len+1);
        
        auto add_point = [&](const int px, const int py) {
          if (px >= 0 && py < output_window.Y && todo.get({px, py}))
            new_points.push_back({px, py});
        };
        
        for (int d=0; d<square_len; d++) {
          add_point(x - square_len, y + d);
          add_point(x - d, y + square_len);
        }
        add_point(x - square_len, y + square_len);
        
        // If there are no new points, rect_strategy is complete
        if (new_points.empty())
          break;
        
        // Check if all points in new_points may be added to current_domain
        // without exceeding result_storage_limit
        for (const auto& p: new_points) {
          current_cost += calculateCost(p, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
          setZeroCost(p, CoefficientIndices, input_wave_lattice, probe_lattice, zero_cost, param);
        }
        
        if (current_cost > result_storage_limit)
          break;
        
        // Add new_points to the current domain
        for (const auto& p: new_points) {
          current_domain.push_back(p);
          todo.set(p, false);
          --remaining_pixels;
        }
        
        // Continue with a larger square in the next iteration
        ++square_len;
      }
      
      // Add current_domain as a new computation domain to rect_strategy
      rect_strategy.computation_domains.push_back(current_domain);
    }
    
    rect_strategy.calculateTotalCost(CoefficientIndices, input_wave_lattice, probe_lattice, output_window, param);
    
    if (verboseLMASearch)
      std::cerr << " (cost: " << rect_strategy.num_multislice_computations << ")";
    
    // Save all computed rectangular strategies (for debugging purposes)
    // rect_strategy.saveAsImage(output_window, std::to_string(width), insert_directory_here);
    
    // Update the best known strategy if the last strategy is better than
    // the previously known best strategy
    if (width == 1) {
      best_strategy = rect_strategy;
    } else {
      if (rect_strategy.num_multislice_computations < best_strategy.num_multislice_computations) {
        if (verboseLMASearch)
          std::cerr << " <- new best strategy";
        
        best_strategy = rect_strategy;
      }
    }
              
    if (verboseLMASearch)
      std::cerr << std::endl;
  }
  
  if (verboseLMASearch)
    std::cerr << "\t\tNumber of computation domains of the best strategy " << best_strategy.computation_domains.size() << std::endl;
  else
    std::cerr << std::endl;
  
  best_strategy.name = "Rectangular";
  return best_strategy;
}

// Chooses a strategy for the Lattice Multislice Algorithm depending on
// the memory requirements. This will be one of the following strategies:
//   - trivialLMAStrategy: compute all probe positions at once
//   - rowLMAStrategy: compute a certain number of rows of probe positions in each step
//   - greedyLMAStrategy: compute sets of probe positions in each step
//   - rectLMAStrategy: compute rectangular sets of probe positions in each step
LMAStrategy chooseLMAStrategy(const OutputWindow& output_window,
                              const Image<std::vector<std::array<int, 2>>>& CoefficientIndices,
                              const Lattice& input_wave_lattice,
                              const Lattice& probe_lattice,
                              const int result_storage_limit,
                              const int lma_strategy_computation_level,
                              const bool writeLMAStrategySearchOutput,
                              const bool verboseLMASearch,
                              const std::string& outputDir,
                              const Param& p) {
  std::cerr << "Choosing a strategy for the Lattice Multislice Algorithm ..." << std::endl;
  
  // The trivial strategy must be used for the Fourier space dirac input wave type
  if (p.inputwave == MS_InputWave::FourierSpaceDirac)
  	return trivialLMAStrategy(output_window,
  	                          CoefficientIndices,
  	                          input_wave_lattice,
  	                          probe_lattice,
  	                          p);
  
  // Count all input waves required for the computation of all probe
  // positions in the output window
  auto cmp = [&](const std::array<int, 2>& a, const std::array<int, 2>& b) {
    return (a[0] + a[1] * input_wave_lattice.X < b[0] + b[1] * input_wave_lattice.X);
  };
  
  std::set<std::array<int, 2>, decltype(cmp)> iw_positions(cmp);
  
  int c = 0;
  for (int y=output_window.start_y; y<output_window.start_y + output_window.Y; y++) {
    for (int x=output_window.start_x; x<output_window.start_x + output_window.X; x++) {
      std::cerr << "\tCounting the required input waves ... "
                << std::setw(5) << std::setprecision(4)
                << (100. * (c++)) / (output_window.X * output_window.Y) << "%     \r";
      
      const int i = x % CoefficientIndices.getX();
      const int j = y % CoefficientIndices.getY();
      
      for (std::array<int, 2> coeff_coord: CoefficientIndices(i, j)) {
        std::array<int, 2> iw_coord = getInputWaveCoord({x, y}, coeff_coord, input_wave_lattice, probe_lattice, p);
        
        iw_positions.insert(iw_coord);
      }
    }
  }
  std::cerr << "\tCounting the required input waves ... 100%       " << std::endl;
  
  std::cerr << std::endl
            << "\t---------------------------------------------------" << std::endl
            << "\t  Minimum number of Multislice calls: " << iw_positions.size() << std::endl
            << "\t---------------------------------------------------" << std::endl
            << std::endl;
  
  // If all input waves required for all probe positions in the output
  // window fit in computer memory without exceeding the memory limit,
  // the simplest strategy of calculating all input waves at once is
  // chosen
  if (static_cast<int>(iw_positions.size()) <= result_storage_limit) {
    std::cerr << "\tSelected trivial strategy (Multislice calls: " << iw_positions.size() << ")" << std::endl << std::endl;
    return trivialLMAStrategy(output_window,
                              CoefficientIndices,
                              input_wave_lattice,
                              probe_lattice,
                              p);
  }
  
  // Search for a good strategy using the greedy algorithm in greedyLMAStrategy() 
  std::cerr << "\tUsing a greedy algorithm to find a good partition of the output window ..." << std::endl;
  LMAStrategy greedy_lma_strategy = greedyLMAStrategy(output_window,
                                                      CoefficientIndices,
                                                      input_wave_lattice,
                                                      probe_lattice,
                                                      result_storage_limit,
                                                      lma_strategy_computation_level,
                                                      verboseLMASearch,
                                                      p);
  
  if (writeLMAStrategySearchOutput)
    greedy_lma_strategy.saveAsImage(output_window, "GreedyAlgorithm", outputDir + "/LMAStrategy");
  
  // Generate the row LMA strategy
  std::cerr << "\tCreating simple row strategy ..." << std::endl;
  LMAStrategy row_lma_strategy = rowLMAStrategy(output_window,
                                                CoefficientIndices,
                                                input_wave_lattice, 
                                                probe_lattice,
                                                result_storage_limit,
                                                verboseLMASearch,
                                                p);
  
  if (writeLMAStrategySearchOutput)
    row_lma_strategy.saveAsImage(output_window, "RowStrategy", outputDir + "/LMAStrategy");
  
  // Generate the rectangle LMA strategy
  std::cerr << "\tCreating rectangular subset strategy ..." << std::endl;
  LMAStrategy rect_lma_strategy = rectLMAStrategy(output_window,
                                                  CoefficientIndices,
                                                  input_wave_lattice, 
                                                  probe_lattice,
                                                  result_storage_limit,
                                                  verboseLMASearch,
                                                  p);
  
  if (writeLMAStrategySearchOutput)
    rect_lma_strategy.saveAsImage(output_window, "RectangleStrategy", outputDir + "/LMAStrategy");
  
  // Print summary and return the cheapest strategy
  std::cerr << std::endl
            << "\tSummary:" << std::endl
            << std::endl
            << "\t    Strategy  |  Cost (= number of Multislice calls)" << std::endl
            << "\t--------------+-------------------------------------" << std::endl
            << "\t     Trivial  |  " << "       N/A" << std::endl
            << "\t      Greedy  |  " << std::setw(10) << greedy_lma_strategy.num_multislice_computations << std::endl
            << "\t         Row  |  " << std::setw(10) << row_lma_strategy.num_multislice_computations << std::endl
            << "\t Rectangular  |  " << std::setw(10) << rect_lma_strategy.num_multislice_computations << std::endl
            << std::endl;
  
  if (greedy_lma_strategy.num_multislice_computations < row_lma_strategy.num_multislice_computations &&
      greedy_lma_strategy.num_multislice_computations < rect_lma_strategy.num_multislice_computations) {
    std::cerr << "\tSelected strategy found by the greedy algorithm." << std::endl << std::endl;
    return greedy_lma_strategy;
  }
  
  if (row_lma_strategy.num_multislice_computations < greedy_lma_strategy.num_multislice_computations &&
      row_lma_strategy.num_multislice_computations < rect_lma_strategy.num_multislice_computations) {
    std::cerr << "\tSelected row strategy." << std::endl << std::endl;
    return row_lma_strategy;
  }
  
  std::cerr << "\tSelected rectangular strategy." << std::endl << std::endl;
  return rect_lma_strategy;
}

#endif  // multislice_lmastrategy_h
