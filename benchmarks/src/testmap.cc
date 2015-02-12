/*
 * map.cc - Map skeleton test.
 *
 * A verbose little bugger who shows a map skeleton "in action".
 */
#include <algorithm>
#include <iostream>

#include <cassert>

#include <SkelCL/SkelCL.h>
#include <SkelCL/Vector.h>
#include <SkelCL/Map.h>
#include <SkelCL/Distributions.h>

#include <pvsutil/Logger.h>

#define VECTOR_SIZE 250000000
#define VECTOR_TYPE int
#define VECTOR_VAL 100

void print(skelcl::Vector<int> A) {
  unsigned int i;
  unsigned int max = std::min(static_cast<unsigned int>(A.size()),
                              static_cast<unsigned int>(20));
  std::cout << "[ ";
  for (i = 0; i < max; i++)
    std::cout << A[i] << " ";
  if (i <= A.size() - 1)
    std::cout << "... ";
  std::cout << "]\n";
}

int main(int argc, char* argv[]) {
  // Turn on logging.
  pvsutil::defaultLogger.setLoggingLevel(pvsutil::Logger::Severity::DebugInfo);

  // Initialise SkelCL to use any device.
  skelcl::init(skelcl::nDevices(1).deviceType(skelcl::device_type::ANY));

  // Define the skeleton objects.
  skelcl::Map<int(int)> map("int func(int x) { return x * 2; }");

  // Define vector input of length "n".
  const int n = VECTOR_SIZE;
  skelcl::Vector<VECTOR_TYPE> input(n);
  std::fill(input.begin(), input.end(), VECTOR_VAL);

  // Set distribution of input.
  skelcl::distribution::setSingle(input);
  input.createDeviceBuffers();

  TIME(upload,   input.copyDataToDevices());
  TIME(exec,     skelcl::Vector<VECTOR_TYPE> output(map(input)));
  TIME(download, output.copyDataToHost());

  print(input);
  print(output);

  return 0;
}
