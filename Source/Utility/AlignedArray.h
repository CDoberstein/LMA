// Container to calculate and store coordinates and size of 2D arrays
// aligned to a given underlying grid
#ifndef multislice_utility_alignedarray_h
#define multislice_utility_alignedarray_h

struct AlignedArray {
  // Position of the corresponding probe / input wave for which this
  // aligned array was created, in Angstrom and in terms of the underlying
  // domain (i.e. distance to the top left corner of the simulation window)
  //
  // The aligned array is centered at this position, up to a sub-pixel
  // shift necessary for the alignment to grid_pixel_size
  const std::array<RealType, 2> wave_position;
  
  // Index of the input wave (value in CoefficientIndices)
  const std::array<int, 2> wave_index;
  
  // Size of the aligned array, in pixel
  const std::array<int, 2> aa_size;
  
  // Position of the top left corner of the aligned array, in pixel and
  // given as the distance to the top left corner of the underlying domain;
  // may be negative
  const std::array<int, 2> aa_pos;
  
  // Distance of wave_position to aa_pos, given in Angstrom
  const std::array<RealType, 2> rel_wave_position;
  
  // Grid pixel size in Angstrom
  const std::array<RealType, 2> grid_pixel_size;
  
  AlignedArray(const AlignedArray&) = default;
  
  AlignedArray(const std::array<RealType, 2>& wave_position,
               const std::array<int, 2>& wave_index,
               const std::array<int, 2>& size,
               const std::array<RealType, 2>& grid_pixel_size)
    : wave_position(wave_position),
      wave_index(wave_index),
      aa_size(size),
      aa_pos({ static_cast<int>(wave_position[0] / grid_pixel_size[0]) - aa_size[0] / 2,
               static_cast<int>(wave_position[1] / grid_pixel_size[1]) - aa_size[1] / 2 }),
      rel_wave_position({ wave_position[0] - aa_pos[0] * grid_pixel_size[0],
                          wave_position[1] - aa_pos[1] * grid_pixel_size[1] }),
      grid_pixel_size(grid_pixel_size)
    { }
};

#endif  // multislice_utility_alignedarray_h
