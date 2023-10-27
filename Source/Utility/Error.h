// Implementation of a function for error handling
#ifndef error_h
#define error_h

#include <iostream>
#include <cstdlib>
#include <string>

void Error(const std::string& msg, const std::string& file, const int line) {
  std::cerr << "Error: " << msg << " (in file \"" << file << "\" on line " << line << ")" << std::endl;
  
  std::cin.get();
  std::quick_exit(1);
}

void Error(const std::string& msg) {
  std::cerr << "Error: " << msg << std::endl;
  
  std::cin.get();
  std::quick_exit(1);
}

#endif  // error_h
