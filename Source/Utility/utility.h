// Miscellaneous utility functions
#ifndef multislice_utility_utility_h
#define multislice_utility_utility_h

#include <cmath>
#include <algorithm>
#include <ctime>
#include <string>
#include <sstream>
#include <filesystem>
#include <fstream>

// Returns the unique element of the set {value + len * z | z in Z} within
// the interval [0, len), where Z is the set of all integers
RealType normalize_periodic(const RealType& value, const RealType& len) {
  return value - std::floor(value / len) * len;
}

// Returns the unique integer of the set {value + len * z | z in Z} within
// the interval [0, len-1], where Z is the set of all integers
int normalize_periodic(const int value, const int len) {
  const int vf = static_cast<int>(std::floor(static_cast<RealType>(value) / len));
  
  return value - vf * len;
}

// Returns the minimum squared distance of two points in a + len * Z and b + len * Z,
// where Z is the set of all integers
RealType periodicSquaredDist(RealType a, RealType b, const RealType len) {
  a = normalize_periodic(a, len);
  b = normalize_periodic(b, len);
  
  const RealType d1 = a - b;
  const RealType d2 = (a < b ? a + len - b : b + len - a);
  
  return std::min(d1 * d1, d2 * d2);
}

// Print current local time and date
void printLocalTimeAndDate() {
  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  
  std::cerr << std::endl
            << "\t---------------------------------------------------" << std::endl
            << std::put_time(&tm, "\tCurrent date (D/M/Y) and time: %d/%m/%Y  %H:%M:%S") << std::endl
            << "\t---------------------------------------------------" << std::endl
            << std::endl;
}

// Prints an explanation on how to use the program
void printUsageHint(const std::string argv0) {
  std::cerr << "Usage: " << argv0 << " path-to-parameter-file [p1 v1 p2 v2 ...]\n"
            << "\n"
            << "  path-to-parameter-file path to a file containing values for all\n"
            << "                         parameters (except for the electron\n"
            << "                         wavelength lambda, which is automatically\n"
            << "                         computed from the accelerating voltage)\n"
            << "\n"
            << "       [p1 v1 p2 v2 ...] parameter values can also be given on the\n"
            << "                         command line to overwrite the values from\n"
            << "                         the parameter file. The format is the same\n"
            << "                         as in the parameter file\n";
}

std::string to_string16(const RealType value) {
  std::ostringstream sstream;
  sstream.precision(16);
  
  sstream << value;
  
  return sstream.str();
}

// Clears the contents of a file or creates an empty file if it doesn't
// exist yet
void clear_file(std::string dir, const std::string& filename) {
  if (dir.empty())
    dir = ".";
  else
    std::filesystem::create_directories(dir);
  
  std::ofstream fstream(dir + '/' + filename, std::ofstream::out | std::ofstream::trunc);
  
  fstream.close();
}

// Appends a single line to a file, creating the file and directories if
// they don't exist yet
void append_line_to_file(std::string dir,
                         const std::string& filename,
                         const std::string& line) {
  if (dir.empty())
    dir = ".";
  else
    std::filesystem::create_directories(dir);
  
  std::ofstream fstream(dir + '/' + filename, std::ofstream::out | std::ofstream::app);
  
  fstream << line << std::endl;
  
  fstream.close();
}

#endif  // multislice_utility_utility_h
