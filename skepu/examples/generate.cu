// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

// following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>
#include <ctime>

#include "skepu/vector.h"
#include "skepu/generate.h"

GENERATE_FUNC(lcg_f, int, int, index, seed,
              unsigned int next = seed;
              unsigned int i;
              for(i = 0; i < index; ++i)
{
next = 1103515245*next + 12345;
}
return (next % 10) + 1;
             )

int main()
{
   skepu::Generate<lcg_f> rand(new lcg_f);

   skepu::Vector<int> v0;

   rand.setConstant(time(0));
   rand(50, v0);

   std::cout<<"v0: " <<v0 <<"\n";

   return 0;
}

