// following define to enable/disable OpenMP implmentation to be used
/* #define SKEPU_OPENMP */

// following define to enable/disable OpenCL implmentation to be used
/* #define SKEPU_OPENCL */
// With OpenCL, following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>

#include "skepu/vector.h"
#include "skepu/maparray.h"

ARRAY_FUNC(arr_f, float, a, b,
           int index = (int)b;
           return a[index];
          )

int main()
{
   skepu::MapArray<arr_f> reverse(new arr_f);

   skepu::Vector<float> v0(10);
   skepu::Vector<float> v1(10);
   skepu::Vector<float> r;

   //Sets v0 = 1 2 3 4 5...
   //     v1 = 19 18 17 16...
   for(int i = 0; i < 10; ++i)
   {
      v0[i] = (float)(i+1);
      v1[i] = (float)(10-i-1);
   }

   std::cout<<"v0: " <<v0 <<"\n";
   std::cout<<"v1: " <<v1 <<"\n";

   reverse(v0, v1, r);

   std::cout<<"r: " <<r <<"\n";

   return 0;
}

