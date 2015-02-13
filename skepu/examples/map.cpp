// following define to enable/disable OpenMP implmentation to be used
/* #define SKEPU_OPENMP */

// following define to enable/disable OpenCL implmentation to be used
/* #define SKEPU_OPENCL */
// With OpenCL, following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>

#include "skepu/matrix.h"
#include "skepu/map.h"

UNARY_FUNC(square_f, int, a,
           return a*a;
          )

int main()
{
   skepu::Map<square_f> square(new square_f);

   skepu::Matrix<int> m(5, 5, 3);
   skepu::Matrix<int> r(5, 5);

   square(m,r);

   std::cout<<"r: " << r <<"\n";

   return 0;
}

