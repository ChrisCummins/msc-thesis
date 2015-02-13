// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

// following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>

#include "skepu/vector.h"
#include "skepu/matrix.h"
#include "skepu/reduce.h"

BINARY_FUNC(plus_f, float, a, b,
            return a+b;
           )

int main()
{
   skepu::Reduce<plus_f> globalSum(new plus_f);

   skepu::Vector<float> v0(50, (float)2);

   std::cout<<"v0: " <<v0 <<"\n";

   float r = globalSum(v0);

   std::cout<<"Result: " <<r <<"\n";

   return 0;
}

