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
  const int n = 1e7;
  skelcl::Vector<int> input(n);
  for (int i = 1; i <= n; i++) input[i - 1] = i;

  // Set distribution of input.
  skelcl::distribution::setSingle(input);
  input.createDeviceBuffers();

  print(input);

  TIME(upload,   input.copyDataToDevices());
  TIME(exec,     skelcl::Vector<int> output(map(input)));
  TIME(download, output.copyDataToHost());

  print(output);

  return 0;
}
