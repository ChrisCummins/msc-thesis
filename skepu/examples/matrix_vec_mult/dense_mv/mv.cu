// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

// following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

/*!
 * An example showing Matrix-Vector multiplication using MapArray skeleton.
 */

#include <iostream>

#include "skepu/maparray.h"
#include "skepu/vector.h"
#include "skepu/matrix.h"


#define SIZE 10

ARRAY_FUNC_MATR_BLOCK_WISE(arr, float, a, b, SIZE,
                           float res = 0;
                           for(int i=0; i<SIZE; ++i)
{
res += a[i] * b[i];
}
return res;
                          )

/*!
 * A helper function to calculate dense matrix-vector product. Used to verify that the SkePU output is correct.
 */
template<typename T>
void directMV(skepu::Vector<T> &v, skepu::Matrix<T> &m, skepu::Vector<T> &res)
{
   int rows = m.total_rows();
   int cols = m.total_cols();

   for(int r=0; r<rows; ++r)
   {
      T sum = T();
      for(int i=0; i<cols; ++i)
      {
         sum += m[r*cols+i] * v[i];
      }
      res[r] = sum;
   }
}

int main()
{
   skepu::MapArray<arr> reverse(new arr);

   skepu::Vector<float> v0(SIZE);
   skepu::Matrix<float> m1(SIZE,SIZE);

   m1.randomize(3,9);

   skepu::Vector<float> r(SIZE);
   skepu::Vector<float> r2(SIZE);

   //Sets v0 = 1 2 3 4 5...
   for(int i = 0; i < SIZE; ++i)
   {
      v0[i] = (float)(i+10);
   }

   std::cout<<"v0: " <<v0 <<"\n";
   std::cout<<"m1: " <<m1 <<"\n";

   reverse(v0, m1, r);
   std::cout<<"r: " <<r <<"\n";

   directMV<float>(v0, m1, r2);
   std::cout<<"r2: " <<r2 <<"\n";

   return 0;
}
