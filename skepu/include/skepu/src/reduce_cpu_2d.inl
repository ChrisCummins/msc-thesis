/*! \file reduce_cpu_2d.inl
 *  \brief Contains the definitions of CPU specific member functions for the 2DReduce skeleton.
 */

#include <iostream>
#include <vector>


#include "operator_type.h"

namespace skepu
{



/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on a
 *  input Matrix. Returns a scalar result.
 *  Using the \em CPU as backend.
 *
 *  \param input A matrix on which the 2D reduction will be performed on.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::CPU(Matrix<T>& input)
{
   DEBUG_TEXT_LEVEL1("REDUCE 2D Matrix CPU\n")

   // Make sure we are properly synched with device data
   input.updateHost();

   size_t rows = input.total_rows();
   size_t cols = input.total_cols();

   std::vector<T> tempResult(rows);

   T *data = input.getAddress();

   for(size_t r=0; r<rows; r++)
   {
      tempResult[r] = data[0];

      for(size_t c=1; c<cols; c++)
      {
         tempResult[r] = m_reduceFuncRowWise->CPU(tempResult[r], data[c]);
      }

      data = data+cols;
   }

   T result = tempResult[0];

   for(size_t r=1; r<rows; r++)
   {
      result = m_reduceFuncColWise->CPU(result, tempResult[r]);
   }

   return result;
}




/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on a
 *  input Sparse Matrix. Returns a scalar result.
 *  Using the \em CPU as backend.
 *
 *  \param input A sparse matrix on which the 2D reduction will be performed on.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::CPU(SparseMatrix<T>& input)
{
   DEBUG_TEXT_LEVEL1("REDUCE 2D SparseMatrix CPU\n")

   // Make sure we are properly synched with device data
   input.updateHost();

   size_t rows = input.total_rows();
   size_t cols = input.total_cols();
   size_t nnz = input.total_nnz();

   std::vector<T> tempResult(rows);
   
//    OPEN_FILE("a2.dat");

   for(size_t r=0; r<rows; r++)
   {
      typename SparseMatrix<T>::iterator it = input.begin(r);

      size_t size= it.size();
      if(size>0)
      {
         tempResult[r] = it[0];

         for(size_t c=1; c<size; c++)
         {
            tempResult[r] = m_reduceFuncRowWise->CPU(tempResult[r], it[c]);
         }
//          WRITE_FILE(std::string("r: " + convertToStr<float>(tempResult[r]) + "\n"));
      }
      else
         tempResult[r] = T();
   }

   T result = tempResult[0];

   for(size_t r=1; r<rows; r++)
   {
      result = m_reduceFuncColWise->CPU(result, tempResult[r]);
   }

   return result;
}


}

