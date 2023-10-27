#ifndef multislice_cpu_transmissionfunction_h
#define multislice_cpu_transmissionfunction_h

#include "TransmissionFunction_aux.h"
#include "SimulationWindow.h"
#include "PropagationWindow.h"

#include "../Utility/Param.h"
#include "../Utility/Specimen.h"
#include "../Utility/CircularImage.h"
#include "../Utility/Image.h"
#include "../Utility/Error.h"

#include <omp.h>

#include <vector>
#include <iostream>

struct TransmissionFunction {
  // Single atom transmission functions, i.e. exp(i*sigma*atom_potential)
  std::vector<CircularImage<complex>> atom_transmission_function;
  
  // Indices of all atoms with respect to specimen.atoms sorted by slice
  std::vector<std::vector<int>> atom_indices_by_slice;
  
  // The full transmission function for each slice of the size of the
  // simulation window
  std::vector<Image<complex>> slice_transmission_function;
  
  // Mapping of the atomic numbers occuring in the specimen to indices
  // starting from 0. Example: if the specimen contains atoms with
  // atomic numbers 10, 25 and 33, then
  //   atomic_number_map[10] = 0,
  //   atomic_number_map[25] = 1 and
  //   atomic_number_map[33] = 2.
  // All other atomic numbers are mapped to -1.
  std::vector<int> atomic_number_map;
  
  void init(const Param& p,
            const SimulationWindow& simulation_window,
            const Specimen& specimen) {
    // Get a list of all atomic species in the specimen and calculate
    // a function mapping the atomic numbers to indices starting from 0
    std::vector<int> atomic_numbers = specimen.getAtomicNumbers();
    
    atomic_number_map = std::vector<int>(200, -1);
    for (unsigned int i=0; i<atomic_numbers.size(); i++)
      atomic_number_map[atomic_numbers[i]] = i;
      
    std::cerr << "\tFound " << atomic_numbers.size() << " different atom species: "
              << atomic_numbers[0];
    for (unsigned int i=1; i<atomic_numbers.size(); i++)
      std::cerr << ", " << atomic_numbers[i];
    std::cerr << std::endl;
    
    // High resolution pixel size for the supersampled circular images
    const RealType hires_pixel_size = std::min(simulation_window.pixel_size_x, simulation_window.pixel_size_y) / p.hires_supersampling_factor;
    
    // Calculate single atom potentials and transmission functions
    // (the single atom transmission functions are exp(i*sigma*atom_potential))
    std::cerr << "\tCalculating single atom potential and transmission functions ..." << std::endl;
    
    std::vector<CircularImage<RealType>> atom_potential;
    atom_transmission_function.clear();
    
    atom_potential.reserve(atomic_numbers.size());
    atom_transmission_function.reserve(atomic_numbers.size());
  
    for (const int Z: atomic_numbers)
      atom_potential.push_back(getProjectedAtomicPotential(Z,
                                                           p.potential_bound,
                                                           hires_pixel_size,
                                                           (p.hires_supersampling_factor/2)*hires_pixel_size));
    
    for (const CircularImage<RealType>& pot: atom_potential)
      atom_transmission_function.push_back(getAtomicTransmissionFunction(pot, p));
    
    // Assign each atom to a slice
    //
    // Note: in the current implementation, atoms are not split across multiple slices and
    //       their entire projected potential is always contained within a single slice
    std::cerr << "\tAssigning each atom to a slice ..." << std::endl;
    
    atom_indices_by_slice = assignAtomsToSlices(specimen, p.slice_thickness);
    
    // Convert the circular atom potential images to rectangular images
    std::vector<Image<RealType>> atom_potential2D(atom_potential.size());
    for (unsigned int i=0; i<atom_potential.size(); i++) {
      atom_potential2D[i] = atom_potential[i].get2DImageNonSquare(simulation_window.pixel_size_x,
                                                                  simulation_window.pixel_size_y,
                                                                  0,
                                                                  p.potential_bound);
      
      if (p.writeInitData) {
        save(atom_potential2D[i], std::to_string(atomic_numbers[i]), p.outputDir + "/MSData/AtomPotentials");
        
        Image<complex> tmp = atom_transmission_function[i].get2DImageNonSquare(simulation_window.pixel_size_x,
                                                                               simulation_window.pixel_size_y,
                                                                               {1, 0},
                                                                               p.potential_bound);
        save(tmp, std::to_string(atomic_numbers[i]), p.outputDir + "/MSData/AtomTransmissionFunctions");
      }
    }
    
    // Calculate the full transmission function for each slice
    std::cerr << "\tCalculating the full transmission function for each slice ..." << std::endl;
    
    slice_transmission_function.resize(atom_indices_by_slice.size());
    
    int progress = 0;
    #pragma omp parallel for num_threads(p.max_num_threads)
    for (int i=0; i<static_cast<int>(atom_indices_by_slice.size()); i++) {
      // Calculate the projected potential for the entire slice
      Image<RealType> slice_pot = getProjectedPotentialSlice(atom_potential2D,
                                                             specimen,
                                                             atom_indices_by_slice[i],
                                                             atomic_number_map,
                                                             simulation_window.pixel_size_x,  // = propagation_window.pixel_size_x
                                                             simulation_window.pixel_size_y,  // = propagation_window.pixel_size_y
                                                             simulation_window.X,
                                                             simulation_window.Y);
      
      // Compute the transmission function
      slice_transmission_function[i] = getSliceTransmissionFunction(slice_pot,
                                                                    p,
                                                                    simulation_window.lenX,
                                                                    simulation_window.lenY);
      
      #pragma omp critical (full_transmission_function_progress)
      {
        std::cerr << "\t\t" << ++progress << " / " << atom_indices_by_slice.size() << "\r";
        
        if (p.writeInitData && p.writeSlices) {
          save(slice_pot, std::to_string(i), p.outputDir + "/MSData/SlicePotential");
          save(slice_transmission_function[i],
               std::to_string(i),
               p.outputDir + "/MSData/SliceTransmissionFunctions/Re",
               p.outputDir + "/MSData/SliceTransmissionFunctions/Im");
        }
      }
    }
    std::cerr << std::endl;
  }
};

#endif  // multislice_cpu_transmissionfunction_h
