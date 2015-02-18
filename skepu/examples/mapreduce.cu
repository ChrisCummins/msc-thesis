// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

// following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>

#include "skepu/vector.h"
#include "skepu/mapreduce.h"

// User-function used for mapping
BINARY_FUNC(mult_f, float, a, b,
            return a*b;
           )

// User-function used for reduction
BINARY_FUNC(plus_f, float, a, b,
            return a+b;
           )

int main()
{
   skepu::MapReduce<mult_f, plus_f> dotProduct(new mult_f, new plus_f);

   skepu::Vector<float> v0(20, (float)2);
   skepu::Vector<float> v1(20, (float)5);

   std::cout<<"v0: " <<v0 <<"\n";
   std::cout<<"v1: " <<v1 <<"\n";

   float r = dotProduct(v0, v1);

   std::cout<<"r: " <<r <<"\n";

   return 0;
}

