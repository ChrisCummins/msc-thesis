// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

// following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */


/*!
 * An example showing SparseMatrix-Vector multiplication using MapArray skeleton.
 */

#include <iostream>

#include "skepu/maparray.h"
#include "skepu/vector.h"
#include "skepu/sparse_matrix.h"


#define SIZE 10


ARRAY_FUNC_SPARSE_MATR_BLOCK_WISE(arr, float, a, b, nnz, aIdx, SIZE,
                                  float res = 0;
                                  for(int i=0; i<nnz; ++i)
{
res += a[aIdx[i]] * b[i];
}
return res;
                                 )


/*!
 * A helper function to calculate SparseMatrix-Vector product. Used to verify that the SkePU output is correct.
 */
template<typename T>
void directspmv(skepu::Vector<T> &v, skepu::SparseMatrix<T> &m, skepu::Vector<T> &res)
{
   int rows = m.total_rows();
   int nnz = m.total_nnz();

   T *values= m.get_values();
   unsigned int * row_offsets = m.get_row_pointers();
   unsigned int * col_indices = m.get_col_indices();

   T sum;

   int rowIdx = 0;
   int nxtRowIdx = 0;

   for(int ii = 0; ii < rows; ii++)
   {
      sum = 0;

      rowIdx = row_offsets[ii];
      nxtRowIdx = row_offsets[ii+1];

      for(int jj=rowIdx; jj<nxtRowIdx; jj++)
      {
         sum += values[jj] * v[col_indices[jj]];
      }
      res[ii] = sum;
   }
}

int main()
{
   skepu::MapArray<arr> reverse(new arr);

   skepu::Vector<float> v0(SIZE);

   // randomly initialize the SparseMatrix
   skepu::SparseMatrix<float> m1(SIZE,SIZE,((SIZE*SIZE)/3), 3.0f, 7.0f);

   // result vectors
   skepu::Vector<float> r(SIZE);
   skepu::Vector<float> r2(SIZE);

   //Sets v0 = 1 2 3 4 5...
   for(int i = 0; i < SIZE; ++i)
   {
      v0[i] = (float)(i+10);
   }

   std::cout<<"v0: " <<v0 <<"\n";
   m1.printMatrixInDenseFormat();

   reverse(v0, m1, r);
   std::cout<<"Computed output: " <<r <<"\n";

   directspmv<float>(v0, m1, r2);
   std::cout<<"Direct output:   " <<r2 <<"\n";

   return 0;
}
