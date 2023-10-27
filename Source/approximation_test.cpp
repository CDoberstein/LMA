// Program to test the approximation of the probe with different values
// of approximation_max_input_waves. In addition, the effect of different
// approximation errors for the probe on the approximation errors in the
// final STEM image is analyzed by running the STEM image simulation for
// each of the approximations and comparing the result to a "ground truth"
// computed in the first step below using MS_InputWave::Probe as input
// waves.
typedef double RealType;

#include "LMA/LatticeMultisliceAlgorithm.h"

int main(int argc, char **argv) {
  if (argc == 1) {
    printUsageHint(argv[0]);
    
    std::cerr << std::endl
              << "Remark: the parameters \"test_max_num_input_waves_start\" and" << std::endl
              << "        \"test_max_num_input_waves_decrement\" need to be set" << std::endl
              << "        for the probe approximation test." << std::endl;
    
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
  
  // Various checks
  if (p.test_max_num_input_waves_start <= 0 || p.test_max_num_input_waves_decrement <= 0)
    Error("Invalid values of the \"test_max_num_input_waves_start\" and "
          "\"test_max_num_input_waves_decrement\" parameters or parameters "
          "not set. Both values must be integers greater than 0 and are "
          "not optional for the probe approximation test.");
  
  if (p.max_probe_approximation_error > 0)
    std::cerr << std::endl
              << "WARNING: the probe approximation error should be zero for the " << std::endl
              << "         approximation test, but it is positive." << std::endl
              << std::endl;
  
  if (p.inputwave == MS_InputWave::Probe || p.inputwave == MS_InputWave::Pixel || p.inputwave == MS_InputWave::FourierSpaceDirac)
    Error("Input wave type set to \"Probe\", \"Pixel\" or \"FourierSpaceDirac\" in the probe approximation "
          "test. Another input wave type should be used.");
  
  if (p.recomputation_count != 0)
    std::cerr << std::endl
              << "WARNING: The recomputation_count parameter is ignored in the " << std::endl
              << "         approximation test." << std::endl
              << std::endl;
  
  // Compute a reference STEM image with the standard Multislice STEM algorithm
  // using the probe itself as the input waves with no approximation
  std::cerr << std::endl
            << "###############################################################" << std::endl
            << std::endl
            << "  Computing reference STEM image with the standard algorithm   " << std::endl
            << std::endl
            << "###############################################################" << std::endl
            << std::endl;
  
  Param p_ref(p);
  p_ref.inputwave = MS_InputWave::Probe;
  
  if (p_ref.frozen_phonon_iterations > 1)
    // Perform ten times more frozen phonon iterations for the reference image than for the
    // images with different probe approximations below
    p_ref.frozen_phonon_iterations *= 10;
  
  LatticeMultisliceAlgorithm LatticeMS_ref(p_ref, true);
  LatticeMS_ref.Compute3DSTEM();
  LatticeMS_ref.clearMultisliceResults();
  
  if (p.save3DSTEM)
    LatticeMS_ref.save3D(p.outputDir + "/Result/3D");
  
  LatticeMS_ref.saveIntegrated2DImages(p.outputDir + "/Result/2D");
  
  LatticeMS_ref.saveNormChange(p.outputDir + "/Result/ExitWaveNormChange");
  
  // Compute STEM images with different numbers of input waves in the
  // probe approximation and compare results with LatticeMS_ref
  std::cerr << std::endl
            << "###############################################################" << std::endl
            << std::endl
            << "   Computing STEM images with different probe approximations   " << std::endl
            << std::endl
            << "###############################################################" << std::endl
            << std::endl;
  
  const std::string dir = p.outputDir + "/ApproximationTest";
  
  clear_file(dir, "bf_euc");
  clear_file(dir, "bf_sup");
  clear_file(dir, "adf_euc");
  clear_file(dir, "adf_sup");
  clear_file(dir, "haadf_euc");
  clear_file(dir, "haadf_sup");
  clear_file(dir, "3d_euc");
  clear_file(dir, "3d_sup");
  clear_file(dir, "approximation_errors_euc");
  clear_file(dir, "approximation_errors_sup");
  
  LatticeMultisliceAlgorithm LatticeMS(p);
  
  for (int n=p.test_max_num_input_waves_start; n>=1; n-=p.test_max_num_input_waves_decrement) {
    // Update the probe approximation to use at most n input waves for
    // the approximation of one probe
    Image<RealType> approximation_errors_euc;
    Image<RealType> approximation_errors_sup;
    
    LatticeMS.updateProbeApproximation(n, approximation_errors_euc, approximation_errors_sup);
    
    // Perform the Lattice Multislice Algorithm to compute a 3D STEM image
    //
    // Note: if partial results are generated, they will always be written
    //       to p.outputDir + "/PartialResult", regardless of n
    LatticeMS.Compute3DSTEM();
    
    // Save the results
    const std::string res_dir = dir + "/Results/" + std::to_string(n);
    
    if (p.save3DSTEM)
      LatticeMS.save3D(res_dir + "/3D");
    
    LatticeMS.saveIntegrated2DImages(res_dir + "/2D");
    
    LatticeMS.saveNormChange(res_dir + "/ExitWaveNormChange");
    
    // Compare result with LatticeMS_ref
    RealType diff3d_euc;
    RealType diff3d_sup;
    std::array<RealType, 3> diff2d_euc;
    std::array<RealType, 3> diff2d_sup;
    std::array<Image<RealType>, 3> diff2d_images;
    
    LatticeMS_ref.calculateRelativeError(LatticeMS,
                                         diff3d_euc, diff3d_sup,
                                         diff2d_euc, diff2d_sup,
                                         diff2d_images);
    
    // Save results to textfiles and as images
    append_line_to_file(dir, "3d_euc", std::to_string(n) + ' ' + to_string16(diff3d_euc));
    append_line_to_file(dir, "3d_sup", std::to_string(n) + ' ' + to_string16(diff3d_sup));
    
    append_line_to_file(dir, "bf_euc", std::to_string(n) + ' ' + to_string16(diff2d_euc[0]));
    append_line_to_file(dir, "bf_sup", std::to_string(n) + ' ' + to_string16(diff2d_sup[0]));
    append_line_to_file(dir, "adf_euc", std::to_string(n) + ' ' + to_string16(diff2d_euc[1]));
    append_line_to_file(dir, "adf_sup", std::to_string(n) + ' ' + to_string16(diff2d_sup[1]));
    append_line_to_file(dir, "haadf_euc", std::to_string(n) + ' ' + to_string16(diff2d_euc[2]));
    append_line_to_file(dir, "haadf_sup", std::to_string(n) + ' ' + to_string16(diff2d_sup[2]));
    
    save(diff2d_images[0], "BF", res_dir + "/2D_diff");
    save(diff2d_images[1], "ADF", res_dir + "/2D_diff");
    save(diff2d_images[2], "HAADF", res_dir + "/2D_diff");
    
    std::string approximation_errors_euc_str;
    for (int i=0; i<approximation_errors_euc.size(); i++)
      approximation_errors_euc_str += to_string16(approximation_errors_euc[i]) + ' ';
    
    std::string approximation_errors_sup_str;
    for (int i=0; i<approximation_errors_sup.size(); i++)
      approximation_errors_sup_str += to_string16(approximation_errors_sup[i]) + ' ';
    
    append_line_to_file(dir, "approximation_errors_euc", std::to_string(n) + ' ' + approximation_errors_euc_str);
    append_line_to_file(dir, "approximation_errors_sup", std::to_string(n) + ' ' + approximation_errors_sup_str);
  }
  
  // Print local time again when finished
  printLocalTimeAndDate();
  
  return 0;
}
