/*
 Taylor series calculation, natural log(1+x)  sum(1:N) (((-1)^(i+1))/i)*x^i
 */

// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

// following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>

#include "skepu/vector.h"
#include "skepu/mapreduce.h"
#include "skepu/generate.h"

/* unary function with constant that is used for mapping in mapreduce operation */
UNARY_FUNC_CONSTANT(nth_term, float, float, t, x,
                    float temp_x = pow(x, t);
                    return (((int)t)%2==0?-1:1)*temp_x/t;
                   )

/* binary function that is used for reduction in mapreduce operation */
BINARY_FUNC(plus, float, a, b,
            return a+b;
           )

/* generate function that is used for vector initialization,
   here constant 'seed' is specified but not used */
GENERATE_FUNC(lcg_init, float, float, index, seed,
              return index+1;
             )

int main(int argc, char ** argv)
{
   int N = 10000; // problem size will affect precision...

// skeleton definition...
   skepu::MapReduce<nth_term, plus> taylor(new nth_term, new plus);
   skepu::Generate<lcg_init> vec_init(new lcg_init); //best way to initialize(?) as it can be done internally in parallel

// vector...
   skepu::Vector<float> v0(N);

// vector initialization using generate skeleton...
   vec_init.setConstant((float)0);
   vec_init(N, v0);

// main taylor computation using mapreduce skeleton.
   taylor.setConstant(1);
   float result = taylor(v0);

   std::cout<<"result: " <<result <<"\n";

   return 0;
}
