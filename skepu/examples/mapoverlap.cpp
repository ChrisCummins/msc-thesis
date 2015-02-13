// following define to enable/disable OpenMP implmentation to be used
/* #define SKEPU_OPENMP */

// following define to enable/disable OpenCL implmentation to be used
/* #define SKEPU_OPENCL */
// With OpenCL, following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>

#include "skepu/vector.h"
#include "skepu/mapoverlap.h"

OVERLAP_FUNC(over_f, float, 2, a,
             return (a[-2]*4 + a[-1]*2 + a[0]*1 +
                    a[1]*2 + a[2]*4)/5;
            )

int main()
{
   skepu::MapOverlap<over_f> conv(new over_f);

   skepu::Vector<float> v0(10, 10);
   skepu::Vector<float> r(10);

   std::cout<<"v0: " <<v0 <<"\n";

   conv(v0,  r, skepu::CONSTANT, (int)0);

   std::cout<<"r: " <<r <<"\n";

   return 0;
}

