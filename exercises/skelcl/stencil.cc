/*
 * map.cc - Map skeleton test.
 *
 * A verbose little bugger who shows a map skeleton "in action".
 */
#include <algorithm>
#include <iostream>
#include <fstream>

#include <SkelCL/SkelCL.h>  // NOLINT(build/include_order)
#include <SkelCL/Matrix.h>  // NOLINT(build/include_order)
#include <SkelCL/Stencil.h>  // NOLINT(build/include_order)
#include <SkelCL/detail/Padding.h>  // NOLINT(build/include_order)

#include <pvsutil/Logger.h>  // NOLINT(build/include_order)

void print(const skelcl::Matrix<double> &A) {
  std::cout << "[\n";
  for (size_t j = 0; j < A.rowCount(); j++) {
      std::cout << "  ";
      for (size_t i = 0; i < A.columnCount(); i++)
          printf("%3.0f ", A[j][i]);
      std::cout << "\n";
  }
  std::cout << "]\n";
}

int main(int argc, char* argv[]) {
  // Turn on logging.
  // pvsutil::defaultLogger.setLoggingLevel(pvsutil::Logger::Severity::DebugInfo);

  // Initialise SkelCL to use any device.
  skelcl::init(skelcl::nDevices(1).deviceType(skelcl::device_type::ANY));

  // Define the skeleton objects.
  unsigned int north = 3;
  unsigned int south = 3;
  unsigned int east = 3;
  unsigned int west = 3;


  skelcl::Stencil<double(double)> sum(
      std::ifstream("./stencil.cl"),
      north, west, south, east,  // extents
      skelcl::detail::Padding::NEUTRAL,  // padding
      0.0,  // neutral value
      "func");  // user function

  // Define Matrix A.
  const int width = 18;
  const int height = 18;
  skelcl::Matrix<double> A({height, width});

  // Initialise matrix.
  auto i = A.begin();
  double c = 0.0;
  while (i != A.end()) {
      *i = c;
      i++;
      c++;
  }

  print(A);
  skelcl::Matrix<double> B = sum(1, A, 1);
  print(B);

  return 0;
}
