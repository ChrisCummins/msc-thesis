#ifndef EXERCISES_OPENCL_HELLO_H_
#define EXERCISES_OPENCL_HELLO_H_

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <iterator>

#include <utility>

#include <CL/cl.hpp>  // NOLINT(build/include_order)

// The string that to "compute".
const std::string hw("Hello OpenCL!\n");

inline void checkError(cl_int err, const char *name) {
  if (err != CL_SUCCESS) {
    std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
    exit(EXIT_FAILURE);
  }
}

#endif  // EXERCISES_OPENCL_HELLO_H_
