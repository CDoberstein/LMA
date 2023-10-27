// Auxiliary function to save the contents of the InputWave struct
#ifndef multislice_cpu_msdata_aux_h
#define multislice_cpu_msdata_aux_h

#include "../Utility/Param.h"
#include "../Utility/Image.h"
#include "PropagationWindow.h"
#include "InputWave.h"

void saveProbeAndInputWave(const Param& p,
                           const InputWave& input,
                           const PropagationWindow& propagation_window,
                           const std::array<int, 2>& probe_window_size) {
  // Save the probe and, for comparison, the probe computed with
  // getProbeKirkland().
  //
  // The results of the two functions will likely differ, which is due to the
  // size of the buffer zone used in getProbeKirkland(), set as a constant inside
  // the function in InputWave_aux.h. A larger buffer zone results in more
  // accurate results with getProbeKirkland() that are closer to the probe
  // calculated with getPropagatedProbe().
  auto probe_kirkland = getProbeKirkland(p, input.hires_pixel_size);
  
  Image<complex> probe2D = input.probe.get2DImageNonSquare(propagation_window.pixel_size_x,
                                                           propagation_window.pixel_size_y,
                                                           probe_window_size[0],
                                                           probe_window_size[1],
                                                           {0, 0});
  Image<complex> probe2D_kirkland = probe_kirkland.get2DImageNonSquare(propagation_window.pixel_size_x,
                                                                       propagation_window.pixel_size_y,
                                                                       probe_window_size[0],
                                                                       probe_window_size[1],
                                                                       {0, 0});
  
  Image<complex> probe2D_fs(probe2D);
  
      // Removes frequencies from the probe according to the value of f below
      // (Only used to generate images for the publication)
    /*const int f=4;
      
      probe2D.applyInverseFourierShift();   // avoids checkerboard pattern in Fourier space image
      FourierTransform(probe2D, &probe2D_fs);
      probe2D_fs.applyFourierShift();
      
      for (int y=0; y<probe2D_fs.getY(); y++)
        for (int x=0; x<probe2D_fs.getX(); x++) {
          const int mx = probe2D_fs.getX()/2;
          const int my = probe2D_fs.getY()/2;
          
          const int fs_radius = 33;
          const int dx = x - mx;
          const int dy = y - my;
          
          if ((x-mx+mx*f)%f != 0 || (y-my+my*f)%f != 0
             || dx*dx + dy*dy > fs_radius*fs_radius
             )
            probe2D_fs(x, y) = {0, 0};
        }
      
      probe2D_fs.applyInverseFourierShift();
      InvFourierTransform(probe2D_fs, &probe2D);
      probe2D.applyFourierShift();*/
  
  save(probe2D,
       "probe",
       p.outputDir + "/MSData/Probe",
       p.outputDir + "/MSData/Probe");
  save(probe2D_kirkland,
       "probe",
       p.outputDir + "/MSData/Probe/KirklandRef",
       p.outputDir + "/MSData/Probe/KirklandRef");
  
  save(pointwise_norm(probe2D),
       "probe_norm",
       p.outputDir + "/MSData/Probe");
  save(pointwise_norm(probe2D_kirkland),
       "probe_norm",
       p.outputDir + "/MSData/Probe/KirklandRef");
  
  probe2D.applyInverseFourierShift();   // avoids checkerboard pattern in Fourier space image
  FourierTransform(probe2D, &probe2D_fs);
  probe2D_fs.applyFourierShift();
  
  save(probe2D_fs,
       "probe_fs",
       p.outputDir + "/MSData/Probe",
       p.outputDir + "/MSData/Probe");
  
  Image<complex> probe2D_kirkland_fs(probe2D_kirkland);
  probe2D_kirkland.applyInverseFourierShift();    // avoids checkerboard pattern in Fourier space image
  FourierTransform(probe2D_kirkland, &probe2D_kirkland_fs);
  probe2D_kirkland_fs.applyFourierShift();
  
  save(probe2D_kirkland_fs,
       "probe_fs",
       p.outputDir + "/MSData/Probe/KirklandRef",
       p.outputDir + "/MSData/Probe/KirklandRef");
  
  // Save the initial condition (if the FourierSpaceDirac input wave is
  // used, save one exemplary image)
  Image<complex> initial_condition2D;
  if (p.inputwave == MS_InputWave::FourierSpaceDirac)
    initial_condition2D = input.getInitialCondition(propagation_window.pixel_size_x,
																							 		  propagation_window.pixel_size_y,
																										propagation_window.X,
																										propagation_window.Y,
																										2,
																										1);
	else
    initial_condition2D = input.getInitialCondition(propagation_window.pixel_size_x,
																							 		  propagation_window.pixel_size_y,
																										propagation_window.X,
																										propagation_window.Y,
																										propagation_window.pixel_size_x * (propagation_window.X / 2),
																										propagation_window.pixel_size_y * (propagation_window.Y / 2));
  save(initial_condition2D,
       "input_wave",
       p.outputDir + "/MSData/InputWave",
       p.outputDir + "/MSData/InputWave");
  
  save(pointwise_norm(initial_condition2D),
       "input_wave_norm",
       p.outputDir + "/MSData/InputWave");
  
  Image<complex> initial_condition2D_fs(initial_condition2D);
  FourierTransform(initial_condition2D, &initial_condition2D_fs);
  initial_condition2D_fs.applyFourierShift();
  
  save(initial_condition2D_fs,
       "input_wave_fs",
       p.outputDir + "/MSData/InputWave",
       p.outputDir + "/MSData/InputWave");
}

#endif  // multislice_cpu_msdata_aux_h
