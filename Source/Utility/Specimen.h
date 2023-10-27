/**
 * Implementation of the Specimen class
 * 
 * Reads and stores the contents of a .xyz file, which includes
 * 
 * - The specimen dimensions in Angstrom
 * 
 * - All atoms in the specimen given by their position, atomic number and standard
 *   deviation of the random thermal motion
 */
#ifndef multislice_utility_specimen_h
#define multislice_utility_specimen_h

#include "Atom.h"
#include "Image3D.h"
#include "Error.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <random>

void read_xyz(const std::string&, RealType&, RealType&, RealType&, std::vector<Atom>&, const bool);

struct Specimen {
  // Random number generator
  std::mt19937 mt_gen;
  
  // Vector of all atoms
  std::vector<Atom> atoms;
  
  // The tile with index (x, y, z) contains the atoms with indices ranging from
  // tile_start(x, y, z) to tile_end(x, y, z)-1, where the indices refer
  // to the atoms vector above.
  Image3D<int> tile_start;
  Image3D<int> tile_end;
  
  // Specimen dimensions in Angstrom
  RealType lenX, lenY, lenZ;
  
  // Constructor initializing the random number generator
  Specimen() {
    std::random_device rd;
    mt_gen.seed(rd());
  }
  
  // Read a xyz file and store the contents in the atoms vector
  void load(const std::string& xyz_file,
            const int tile_x,
            const int tile_y,
            const int tile_z,
            const bool no_vibration) {
    // Read specimen file
    read_xyz(xyz_file, lenX, lenY, lenZ, atoms, no_vibration);
    
    const int num_atoms = static_cast<int>(atoms.size());
    
    // Print unit cell dimensions and number of atoms
    std::cerr << "Read " << num_atoms << " atoms from file. Original specimen size: "
              << lenX << " x " << lenY << " x " << lenZ << " Angstrom" << std::endl;
    
    // Tiling of the unit cell
    tile_start = Image3D<int>(tile_x, tile_y, tile_z);
    tile_end = Image3D<int>(tile_x, tile_y, tile_z);
    
    atoms.reserve(tile_x*tile_y*tile_z*num_atoms);
    
    int c = 0;
    for (int tx=0; tx<tile_x; tx++)
      for (int ty=0; ty<tile_y; ty++)
        for (int tz=0; tz<tile_z; tz++, c+=num_atoms) {
          tile_start(tx, ty, tz) = c;
          tile_end(tx, ty, tz) = c + num_atoms;
          
          if (tx==0 && ty==0 && tz==0)
            continue;
          
          for (int i=0; i<num_atoms; i++)
            atoms.push_back(Atom(atoms[i].Z,
                                 atoms[i].x + tx*lenX,
                                 atoms[i].y + ty*lenY,
                                 atoms[i].z + tz*lenZ,
                                 atoms[i].thermal_sigma));
        }
    
    lenX *= tile_x;
    lenY *= tile_y;
    lenZ *= tile_z;
    
    // Print the resulting specimen dimensions
    std::cerr << "Specimen dimensions after tiling: " << lenX << " x " << lenY << " x " << lenZ << " Angstrom ("
              << atoms.size() << " atoms)" << std::endl
              << std::endl;
  }
  
  // Returns a vector of all atomic numbers of atoms in the specimen
  std::vector<int> getAtomicNumbers() const {
    std::vector<int> atomic_numbers;
    
    for (const auto& cur_atom: atoms) {
      unsigned int i=0;
      for (; i < atomic_numbers.size(); i++)
        if (atomic_numbers[i] == cur_atom.Z)
          break;
      if (i == atomic_numbers.size())
        atomic_numbers.push_back(cur_atom.Z);
    }
    
    return atomic_numbers;
  }
  
  // Updates one tile of the specimen given by the update_tile_{x|y|z}
  // arguments to the new specimen structure in new_xyz_file and
  // returns a vector of all xy coordinates in Angstrom of locations of
  // changes from the previous specimen structure, including as the third
  // value the maximum value of thermal_sigma of an atom that changed at
  // the given position
  std::vector<std::array<RealType, 3>> update_tile(const std::string& new_xyz_file,
                                                   const int update_tile_x,
                                                   const int update_tile_y,
                                                   const int update_tile_z,
                                                   const bool no_vibration) {
    // Read new specimen unit cell from file
    RealType lenX2, lenY2, lenZ2;
    std::vector<Atom> tile_atoms_new;
    
    read_xyz(new_xyz_file, lenX2, lenY2, lenZ2, tile_atoms_new, no_vibration);
    
    const int tile_x = tile_start.getX();
    const int tile_y = tile_start.getY();
    const int tile_z = tile_start.getZ();
    if (lenX2 * tile_x != lenX || lenY2 * tile_y != lenY || lenZ2 * tile_z != lenZ)
      Error("Incompatible specimen dimensions in file \"" + new_xyz_file + "\"!");
    
    // Shift atoms to the appropriate tile
    for (Atom& atom: tile_atoms_new) {
      atom.x += update_tile_x * lenX2;
      atom.y += update_tile_y * lenY2;
      atom.z += update_tile_z * lenZ2;
    }
    
    // Calculate a list of all atoms that change within the tile that is
    // updated
    const int start = tile_start(update_tile_x, update_tile_y, update_tile_z);
    const int end = tile_end(update_tile_x, update_tile_y, update_tile_z);
    
    std::vector<Atom> tile_atoms_current(atoms.cbegin() + start, atoms.cbegin() + end);
    
    std::sort(tile_atoms_current.begin(), tile_atoms_current.end());
    std::sort(tile_atoms_new.begin(), tile_atoms_new.end());
    
    std::vector<Atom> symdiff;
 
    std::set_symmetric_difference(tile_atoms_current.cbegin(), tile_atoms_current.cend(),
                                  tile_atoms_new.cbegin(), tile_atoms_new.cend(),
                                  std::back_inserter(symdiff));
    
    // Update the tile_start and tile_end arrays by offsetting all indices
    // past the tile where atoms may have been inserted or removed
    const int count_diff = static_cast<int>(tile_atoms_new.size()) - static_cast<int>(tile_atoms_current.size());
    
    for (int i=0; i<tile_start.size(); i++) {
      if (tile_start[i] > start)
        tile_start[i] += count_diff;
    }
    
    for (int i=0; i<tile_end.size(); i++) {
      if (tile_end[i] > start)
        tile_end[i] += count_diff;
    }
    
    // Update the atoms in the given tile
    std::vector<Atom> atoms_new;
    atoms_new.reserve(atoms.size() + count_diff);
    
    atoms_new.insert(atoms_new.end(), atoms.cbegin(), atoms.cbegin() + start);
    atoms_new.insert(atoms_new.end(), tile_atoms_new.cbegin(), tile_atoms_new.cend());
    atoms_new.insert(atoms_new.end(), atoms.cbegin() + end, atoms.cend());
    
    atoms = atoms_new;
    
    // Create a vector of xy locations of changes, removing duplicates
    auto cmp = [](const std::array<RealType, 3>& a, const std::array<RealType, 3>& b) {
      return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
    };
    
    std::set<std::array<RealType, 3>, decltype(cmp)> change_locations(cmp);
    for (const Atom& atom: symdiff)
      change_locations.insert({atom.x, atom.y, atom.thermal_sigma});
    
    // Convert to a vector and only keep the entry with the largest value of thermal_sigma
    std::vector<std::array<RealType, 3>> result;
    result.reserve(change_locations.size());
    
    for (const std::array<RealType, 3>& cl: change_locations) {
      unsigned int i=0;
      for (; i<result.size() && !(result[i][0] == cl[0] && result[i][1] == cl[1]); i++);
      
      if (i < result.size())
        result[i][2] = std::max(result[i][2], cl[2]);
      else
        result.push_back(cl);
    }
    
    return result;
  }
  
  // Randomly displaces all atoms according to their thermal_sigma parameter
  void applyVibration() {
    for (Atom& atom: atoms) {
      std::normal_distribution<RealType> d(0, atom.thermal_sigma);
      
      atom.x = atom.base_x + d(mt_gen);
      atom.y = atom.base_y + d(mt_gen);
      atom.z = atom.base_z + d(mt_gen);
    }
  }
};

// Read a xyz file from file and return the specimen dimensions as well
// as a vector of all atoms in the specimen
void read_xyz(const std::string& xyz_file,
              RealType& lenX,
              RealType& lenY,
              RealType& lenZ,
              std::vector<Atom>& atoms,
              const bool no_vibration) {
  std::ifstream fs(xyz_file);
  if (!fs.is_open())
    Error("Unable to open \"" + xyz_file + "\"!", __FILE__, __LINE__);
  
  std::string line;
  
  // Skip comment line
  std::getline(fs, line);
  
  // Read unit cell dimensions in Angstroms
  fs >> lenX >> lenY >> lenZ;
  
  // Read atom data
  while(fs.good()) {
    int Z;
    RealType x, y, z;
    RealType occ;
    RealType thermal_sigma;
    
    fs >> Z;
    
    if (Z == -1 || !fs.good())
      break;
    
    fs >> x >> y >> z >> occ >> thermal_sigma;
    
    if (no_vibration)
      thermal_sigma = 0;
    
    if (occ != 1.0)
      std::cerr << "Warning: detected occupancy value different from 1. (Occupancy "
                   "values are ignored in the current implementation)" << std::endl;
    
    atoms.push_back(Atom(Z, x, y, z, thermal_sigma));
  }
  
  if (atoms.empty())
    Error("Failed to read atom data in file \"" + xyz_file + "\"!", __FILE__, __LINE__);
}

#endif  // multislice_utility_specimen_h
