typedef double RealType;

#include "LMA/LatticeMultisliceAlgorithm.h"

int main(int argc, char **argv) {
  if (argc == 1) {
    printUsageHint(argv[0]);
    
    return 0;
  }
  
  // Print local time at program startup
  printLocalTimeAndDate();
  
  // Read microscope and simulation settings
  std::cerr << "Reading the parameter file \"" << argv[1] << "\" ..." << std::endl;
  Param p = readParameterFile(argv[1]);
  
  std::string commandline_parameters;
  for (int i=2; i<argc; i++)
    commandline_parameters += std::string(" ") + argv[i];
  
  parseParameterString(commandline_parameters, p, false);
  
  std::cerr << "Electron wavelength = " << p.lambda << " Angstrom (at " << p.AcceleratingVoltage / 1000 << " kV)\n\n";
  
  // Save a copy of the parameters to the output directory
  writeParameterFile(p.outputDir, p.parameterfile_name, p);
  
  // Initialize the data structures for the Lattice Multislice Algorithm
  LatticeMultisliceAlgorithm LatticeMS(p);
  
  // Print local time after initialization
  printLocalTimeAndDate();
  
  // Perform the Lattice Multislice Algorithm to compute a 3D STEM image
  LatticeMS.Compute3DSTEM();
  
  // Save the results
  if (p.save3DSTEM)
    LatticeMS.save3D(p.outputDir + "/Result/3D");
  
  LatticeMS.saveIntegrated2DImages(p.outputDir + "/Result/2D");
  
  LatticeMS.saveNormChange(p.outputDir + "/Result/ExitWaveNormChange");
  
  // Print local time again when finished
  printLocalTimeAndDate();
  
  // Recompute STEM images with local changes if recomputation_count is
  // greater than zero
  for (int i=0; i<p.recomputation_count; i++) {
    LatticeMS.Recompute3DSTEM(i+1);
    
    // Save the results
    const std::string base_dir = p.outputDir + "/Result/LocalChanges/" + std::to_string(i+1);
    
    if (p.save3DSTEM)
      LatticeMS.save3D(base_dir + "/3D");
    
    LatticeMS.saveIntegrated2DImages(base_dir + "/2D");
    
    LatticeMS.saveNormChange(base_dir + "/ExitWaveNormChange");
    
    // Print local time after every iteration
    printLocalTimeAndDate();
  }
  
  return 0;
}
