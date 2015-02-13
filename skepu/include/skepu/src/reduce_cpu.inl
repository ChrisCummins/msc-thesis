/*! \file reduce_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the Reduce skeleton.
 */

#include <iostream>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs the Reduction on a whole Vector. Returns a scalar result. A wrapper for CPU(InputIterator inputBegin, InputIterator inputEnd).
 *  Using the \em CPU as backend.
 *
 *  \param input A vector which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::CPU(Vector<T>& input)
{
   return CPU(input.begin(), input.end());
}


/*!
 *  Performs the Reduction on a whole Matrix. Returns a scalar result. A wrapper for CPU(InputIterator inputBegin, InputIterator inputEnd).
 *  Using the \em CPU as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::CPU(Matrix<T>& input)
{
   return CPU(input.begin(), input.end());
}


/*!
 *  Performs the Reduction on a whole Matrix either row-wise or column-wise. Returns a \em SkePU vector of reduction result.
 *  Using the \em CPU as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \param reducePolicy The policy specifying how reduction will be performed, can be either REDUCE_ROW_WISE_ONLY of REDUCE_COL_WISE_ONLY
 *  \return A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
Vector<T> Reduce<ReduceFunc, ReduceFunc>::CPU(Matrix<T>& input, ReducePolicy reducePolicy)
{
   DEBUG_TEXT_LEVEL1("REDUCE Matrix[] CPU\n")

   Matrix<T> *matrix = NULL;

   if(reducePolicy==REDUCE_COL_WISE_ONLY)
      matrix = &(~input);
   else // assume  reducePolict==REDUCE_ROW_WISE_ONLY)
      matrix = &input;

   // Make sure we are properly synched with device data
   matrix->updateHost();

   size_t rows = matrix->total_rows();
   size_t cols = matrix->total_cols();

   skepu::Vector<T> result(rows);

   T *data = matrix->getAddress();

   for(size_t r=0; r<rows; r++)
   {
      result(r) = data[0];

      for(size_t c=1; c<cols; c++)
      {
         result(r) = m_reduceFunc->CPU(result(r), data[c]);
      }

      data = data+cols;
   }

   return result;
}


/*!
 *  Performs the Reduction on non-zero elements of a SparseMatrix. Returns a scalar result.
 *  Using the \em CPU as backend.
 *
 *  \param input A sparse matrix which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::CPU(SparseMatrix<T>& input)
{
   DEBUG_TEXT_LEVEL1("REDUCE SparseMatrix CPU\n")

   // Make sure we are properly synched with device data
   input.updateHost();

   T * values= input.get_values();
   size_t nnz = input.total_nnz();

   T tempResult = values[0];

   for(size_t i=1; i<nnz; ++i)
   {
      tempResult = m_reduceFunc->CPU(tempResult, values[i]);
   }

   return tempResult;
}



/*!
 *  Performs the Reduction on non-zero elements of a SparseMatrix either row-wise or column-wise. Returns a vector of reduction result.
 *  Using the \em CPU as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \param reducePolicy The policy specifying how reduction will be performed, can be either REDUCE_ROW_WISE_ONLY of REDUCE_COL_WISE_ONLY
 *  \return A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
Vector<T> Reduce<ReduceFunc, ReduceFunc>::CPU(SparseMatrix<T>& input, ReducePolicy reducePolicy)
{
   DEBUG_TEXT_LEVEL1("REDUCE SparseMatrix[] CPU\n")

   SparseMatrix<T> *matrix = NULL;

   if(reducePolicy==REDUCE_COL_WISE_ONLY)
      matrix = &(~input);
   else // assume  reducePolicy==REDUCE_ROW_WISE_ONLY)
      matrix = &input;

   // Make sure we are properly synched with device data
   matrix->updateHost();

   size_t rows = matrix->total_rows();
   size_t cols = matrix->total_cols();
   size_t nnz = matrix->total_nnz();
   
//    OPEN_FILE("a1.dat");

   Vector<T> result(rows);
//    size_t count = 0;
   for(size_t r=0; r<rows; r++)
   {
      typename SparseMatrix<T>::iterator it = matrix->begin(r);

      size_t size= it.size();
      
      if(size>0)
      {
         result[r] = it[0];

         for(size_t c=1; c<size; c++)
         {
            result[r] = m_reduceFunc->CPU(result[r], it[c]);
         }
//          WRITE_FILE(std::string("r: " + convertToStr<float>(result(r)) + "\n"));
//          std::cerr << "size " << r << ": " << size << "\n";
//          count += size;
      }
      else
         result[r] = T();
   }
//    std::cerr << "total: " << count << " == " << nnz << "\n";

   return result;
}




/*!
 *  Performs the Reduction on a range of elements. Returns a scalar result. Does the reduction on the \em CPU
 *  by iterating over all elements in the range.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type Reduce<ReduceFunc, ReduceFunc>::CPU(InputIterator inputBegin, InputIterator inputEnd)
{
   DEBUG_TEXT_LEVEL1("REDUCE CPU\n")

   // Make sure we are properly synched with device data
   inputBegin.getParent().updateHost();

   // Uses operator () to avoid unneccessary synchronization function calls
   typename InputIterator::value_type tempResult = inputBegin(0);
   inputBegin++;
   for(; inputBegin != inputEnd; ++inputBegin)
   {
      tempResult = m_reduceFunc->CPU(tempResult, inputBegin(0));
   }

   return tempResult;
}

}

