// Auxiliary functions for reading parameter values from a string
#ifndef multislice_utility_param_aux_h
#define multislice_utility_param_aux_h

#include "Error.h"

#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include <iomanip>

int get_last_index(const std::vector<std::pair<std::string, std::string>>& name_value,
                   const bool require_necessary_params,
                   const std::string& name,
                   const bool optional_param) {
  int i=static_cast<int>(name_value.size())-1;
  
  for (; i>=0; i--)
    if (name_value[i].first == name)
      return i;
  
  if (require_necessary_params && !optional_param)
    Error("Parameter \"" + name + "\" not set!");
  
  return -1;
}

void i_update_real(const std::string& name,
                   RealType& value,
                   const bool optional_param,
                   const RealType default_value,
                   const std::vector<std::pair<std::string, std::string>>& name_value,
                   const bool require_necessary_params) {
  const int i = get_last_index(name_value, require_necessary_params, name, optional_param);
  
  if (i!=-1) {
    std::stringstream ss(name_value[i].second);
    ss >> value;
  } else if (require_necessary_params && optional_param) {
    value = default_value;
  }
}

void i_update_int(const std::string& name,
                  int& value,
                  const bool optional_param,
                  const int default_value,
                  const std::vector<std::pair<std::string, std::string>>& name_value,
                  const bool require_necessary_params) {
  const int i = get_last_index(name_value, require_necessary_params, name, optional_param);
  
  if (i!=-1) {
    std::stringstream ss(name_value[i].second);
    ss >> value;
  } else if (require_necessary_params && optional_param) {
    value = default_value;
  }
}

void i_update_bool(const std::string& name,
                   bool& value,
                   const bool optional_param,
                   const bool default_value,
                   const std::vector<std::pair<std::string, std::string>>& name_value,
                   const bool require_necessary_params) {
  const int i = get_last_index(name_value, require_necessary_params, name, optional_param);
  
  if (i!=-1) {
    if (name_value[i].second == "false" ||
        name_value[i].second == "False" ||
        name_value[i].second == "0") {
      value = false;
    } else if (name_value[i].second == "true" ||
               name_value[i].second == "True" ||
               name_value[i].second == "1") {
      value = true;
    } else {
      Error("Invalid boolean value \"" + name_value[i].second + "\" given for "
            "parameter \"" + name + "\"!");
    }
  } else if (require_necessary_params && optional_param) {
    value = default_value;
  }
}

bool i_update_string(const std::string& name,
                     std::string& value,
                     const bool optional_param,
                     const std::string default_value,
                     const std::vector<std::pair<std::string, std::string>>& name_value,
                     const bool require_necessary_params) {
  const int i = get_last_index(name_value, require_necessary_params, name, optional_param);
  
  if (i!=-1) {
    value = name_value[i].second;
    return true;
    
  } else if (require_necessary_params && optional_param) {
    value = default_value;
    return true;
  }
  
  return false;
}

#endif  // multislice_utility_param_aux_h
