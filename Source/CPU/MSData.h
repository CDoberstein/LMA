// Implementation of the MSData struct that contains all data required by the Multislice
// algorithm
#ifndef multislice_cpu_msdata_h
#define multislice_cpu_msdata_h

#include "MSData_aux.h"

#include "../Utility/Param.h"
#include "../Utility/Specimen.h"
#include "../Utility/Error.h"
#include "SimulationWindow.h"
#include "PropagationWindow.h"
#include "TransmissionFunction.h"
#include "PropagationFunction.h"
#include "InputWave.h"

#include <iostream>
#include <cmath>
#include <vector>
#include <array>

// Container for the specimen and precomputed data for the MS algorithm
struct MSData {
  // A copy of the simulation parameters used for the generation of the Multislice data
  const Param p;
  
  // The specimen structure
  Specimen specimen;
  
  // Transmission functions for every slice and individual atoms
  TransmissionFunction tf;
  
  // Propagation function
  PropagationFunction pf;
  
  // The input waves for the multislice algorithm and information on how to approximate
  // the STEM probes as linear combinations
  InputWave input;
  
  // Size of the full simulation window covering the entire specimen used in the Fourier
  // space algorithms (i.e. when p.domain == MS_Domain::FourierSpace)
  SimulationWindow simulation_window;
  
  // Size of the small simulation window used in the real space algorithms
  // (i.e. when p.domain == MS_Domain::RealSpace)
  PropagationWindow propagation_window;
  
  // Size of the probe window (i.e. the size of the propagation window
  // if the probes themselves were used as the input waves)
  std::array<int, 2> probe_window_size;
  
  MSData(const Param& p) : p(p) {
    if (p.frozen_phonon_iterations < 1)
      Error("The number of frozen phonon iterations must be positive! (frozen_phonon_iterations parameter)");
    
    // Print a brief overview of the simulation mode
    std::cerr << "Simulation mode" << std::endl
              << "----+-----------------------------------------------" << std::endl
              << "    |     Device: " << (p.device == MS_Device::CPU ? "CPU" :
                                     (p.device == MS_Device::GPU ? "GPU" :
                                                                   "-- unknown --")) << std::endl
              << "    |     Domain: " << (p.domain == MS_Domain::RealSpace ?    "real space" :
                                     (p.domain == MS_Domain::FourierSpace ? "Fourier space" :
                                                                            "-- unknown --")) << std::endl
              << "    | Propagator: " << (p.propagator == MS_Propagator::Full ?                 "standard (full rank)" :
                                         (p.propagator == MS_Propagator::LowRankApproximation ? "low rank approximation" :
                                                                                                "-- unknown --")) << std::endl
              << "    | Input wave: " << (p.inputwave == MS_InputWave::Probe ?                     "Probe wave (no approximation)" :
                                         (p.inputwave == MS_InputWave::ProbeSubset ?               "Probe wave (with approximation)" :
                                         (p.inputwave == MS_InputWave::TrigonometricPolynomial ?   "Trigonometric polynomial" :
                                         (p.inputwave == MS_InputWave::TrigonometricPolynomialV2 ? "Trigonometric polynomial (V2)" :
                                         (p.inputwave == MS_InputWave::Square ?                    "Square" :
                                         (p.inputwave == MS_InputWave::Disk ?                      "Disk" :
                                         (p.inputwave == MS_InputWave::Pixel ?                     "Pixel" :
                                         (p.inputwave == MS_InputWave::Gaussian ?                  "Gaussian" :
                                         (p.inputwave == MS_InputWave::FourierSpaceDirac ?         "Fourier space Dirac deltas (PRISM)" :
                                                                                                   "-- unknown --"))))))))) << std::endl
              << "----+-----------------------------------------------" << std::endl
              << std::endl;
    
    // Load the specimen structure from a .xyz file
    std::cerr << "Loading the specimen from file ..." << std::endl;
    specimen.load(p.specimen_path, p.tile_x, p.tile_y, p.tile_z, p.frozen_phonon_iterations == 1);
    
    // Calculate the size of the full simulation window
    std::cerr << "Calculating the size of the full simulation window ..." << std::endl;
    simulation_window.init(p, specimen);
    
    std::cerr << "\t      Simulation window size: " << simulation_window.X << " x " << simulation_window.Y << " (pixel)" << std::endl
              << "\t                              " << simulation_window.lenX << " x " << simulation_window.lenY << " (Angstrom)" << std::endl
              << "\tSimulation window pixel size: " << simulation_window.pixel_size_x << " x " << simulation_window.pixel_size_y << " (Angstrom)" << std::endl
              << std::endl;
    
    // Calculate the input wave functions to the multislice algorithm (i.e. the initial
    // conditions of the Schrödinger equation)
    std::cerr << "Initializing the input wave functions ..." << std::endl;
    input.init(p, simulation_window);
    std::cerr << std::endl;
    
    // Calculate the size of the (small) propagation window for the realspace algorithms
    std::cerr << "Calculating the size of the propagation window ..." << std::endl;
    propagation_window.init(p, input, simulation_window);
    
    std::cerr << "\t      Propagation window size: " << propagation_window.X << " x " << propagation_window.Y << " (pixel)" << std::endl
              << "\t                               " << propagation_window.lenX << " x " << propagation_window.lenY << " (Angstrom)" << std::endl
              << "\tPropagation window pixel size: " << propagation_window.pixel_size_x << " x " << propagation_window.pixel_size_y << " (Angstrom)" << std::endl
              << std::endl;
    
    if (p.fs_buffer_zone >= 0 && p.domain == MS_Domain::FourierSpace)
      std::cerr << "\tUsing extended propagation window size: " << propagation_window.ext_X << " x " << propagation_window.ext_Y << " (pixel)" << std::endl
                << "\t                                        " << propagation_window.ext_lenX << " x " << propagation_window.ext_lenY << " (Angstrom)" << std::endl
                << std::endl;
    
    // Calculate the probe window size
    probe_window_size = {static_cast<int>(std::ceil(2 * p.probe_radius / propagation_window.pixel_size_x)),
                         static_cast<int>(std::ceil(2 * p.probe_radius / propagation_window.pixel_size_y))};
    
    if (p.inputwave == MS_InputWave::FourierSpaceDirac)
      // For the Fourier space dirac input waves, the probe window size must
      // be at most the simulation window size divided by p.input_wave_lattice_r
      // due to the implicit probe repetition for this type of approximation
      // of the STEM probes
      // (This also has the side-effect of ensuring that the probe window does
      //  not exceed the simulation window in size)
      probe_window_size = {std::min(probe_window_size[0], simulation_window.X / p.input_wave_lattice_r),
				                   std::min(probe_window_size[1], simulation_window.Y / p.input_wave_lattice_r)};
    
    // Save the probe and input wave functions to file
    if (p.writeInitData) {
      std::cerr << "Saving the probe and input wave functions ..." << std::endl;
      saveProbeAndInputWave(p, input, propagation_window, probe_window_size);
      std::cerr << std::endl;
    }
    
    // Calculate the transmission functions
    std::cerr << "Calculating the transmission functions ..." << std::endl;
    tf.init(p, simulation_window, specimen);
    std::cerr << std::endl;
    
    // Calculate the propagation function
    std::cerr << "Calculating the propagation function ..." << std::endl;
    pf.init(p, simulation_window, propagation_window);
    std::cerr << std::endl;
  }
  
  // Updates the transmission functions to the n-th specimen *.xyz file
  // for the recomputation of local changes (see also the explanation of
  // the recomputation_count parameter in Utility/Param.h)
  std::vector<std::array<RealType, 3>> updateTransmissionFunctions(const int n) {
    // Generate the new specimen structure
    std::cerr << "\tReading updated specimen structure from file ";
    
    const std::string xyz_path = (n <= 0 ? p.specimen_path : p.specimen_path.substr(0, p.specimen_path.length()-4) + std::to_string(n) + ".xyz");
    
    std::cerr << "(" << xyz_path << ") ..." << std::endl;
    
    auto change_locations = specimen.update_tile(xyz_path,
                                                 p.recomputation_change_tile_x,
                                                 p.recomputation_change_tile_y,
                                                 p.recomputation_change_tile_z,
                                                 p.frozen_phonon_iterations == 1);
    
    // Update the transmission functions
    tf.init(p, simulation_window, specimen);
    
    return change_locations;
  }
  
  // Randomly shifts all atoms in the specimen according to their thermal_sigma
  // parameter and updates the transmission functions
  void nextFrozenPhononState() {
    // Displace atoms randomly
    specimen.applyVibration();
    
    // Update the transmission function
    tf.init(p, simulation_window, specimen);
  }
};

#endif  // multislice_cpu_msdata_h
