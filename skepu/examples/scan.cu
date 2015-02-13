// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

// following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>

#include "skepu/vector.h"
#include "skepu/scan.h"

BINARY_FUNC(plus_f, float, a, b,
            return a+b;
           )

int main()
{
   skepu::Scan<plus_f> prefix_sum(new plus_f);

   skepu::Vector<float> v0(50, (float)1);
   skepu::Vector<float> r;

   std::cout<<"v0: " <<v0 <<"\n";

   prefix_sum(v0, r, skepu::INCLUSIVE, (float)10);

   std::cout<<"r: " <<r <<"\n";

   return 0;
}

