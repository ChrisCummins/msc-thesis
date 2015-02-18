/*! \file reduce_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the Reduce skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>
#include <iostream>
#include <vector>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs the Reduction on a whole Vector. Returns a scalar result. A wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd).
 *  Using \em OpenMP as backend.
 *
 *  \param input A vector which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::OMP(Vector<T>& input)
{
   return OMP(input.begin(), input.end());
}


/*!
 *  Performs the Reduction on a whole Matrix. Returns a scalar result. A wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd).
 *  Using \em OpenMP as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::OMP(Matrix<T>& input)
{
   return OMP(input.begin(), input.end());
}



/*!
 *  Performs the Reduction on a whole Matrix. Returns a \em SkePU vector of reduction result.
 *  Using \em OpenMP as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \param reducePolicy The policy specifying how reduction will be performed, can be either REDUCE_ROW_WISE_ONLY of REDUCE_COL_WISE_ONLY
 *  \return A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
skepu::Vector<T> Reduce<ReduceFunc, ReduceFunc>::OMP(Matrix<T>& input, ReducePolicy reducePolicy)
{
   DEBUG_TEXT_LEVEL1("REDUCE Matrix[] OpenMP\n")

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

   size_t totElem = rows*cols;

   omp_set_num_threads(m_execPlan->numOmpThreads(totElem));

   unsigned int numThreads = omp_get_max_threads();

   if(numThreads>rows) // when number of threads is less than processed rows, set 1 thread per row.
   {
      numThreads = rows;
      omp_set_num_threads(numThreads);
   }

   // Now, we safely assume that there are at least as number of rows to process as #threads available

   // schedule rows to each thread
   size_t rowsPerThread=rows/numThreads;

   size_t restRows = rows%numThreads;

   unsigned int myid;
   size_t firstRow, lastRow;

   T psum;

   // we divide the "N" remainder rows to first "N" threads instead of giving it to last thread to achieve better load balancing
   #pragma omp parallel private(myid, firstRow, lastRow, psum) default(shared)
   {
      myid = omp_get_thread_num();

      firstRow = myid*rowsPerThread;

      if(myid!=0)
         firstRow += (myid<restRows)? myid:restRows;

      if(myid < restRows)
         lastRow = firstRow+rowsPerThread+1;
      else
         lastRow = firstRow+rowsPerThread;

      for(size_t r= firstRow; r<lastRow; ++r)
      {
         size_t base = r*cols;
         psum = data[base];
         for(size_t c=1; c<cols; ++c)
         {
            psum = m_reduceFunc->CPU(psum, data[base+c]);
         }
         result(r) = psum;
      }
   }

   return result;
}





/*!
 *  Performs the Reduction on non-zero elements of a SparseMatrix. Returns a scalar result.
 *  Using \em OpenMP as backend.
 *
 *  \param input A sparse matrix which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::OMP(SparseMatrix<T>& input)
{
   DEBUG_TEXT_LEVEL1("REDUCE SparseMatrix OpenMP\n")

   // Make sure we are properly synched with device data
   input.updateHost();

   T * values= input.get_values();
   size_t nnz = input.total_nnz();

   omp_set_num_threads(m_execPlan->numOmpThreads(nnz));

   unsigned int nthr = omp_get_max_threads();
   size_t q = nnz/nthr;
   size_t rest = nnz%nthr;
   unsigned int myid;
   size_t first, last;
   T psum;

   if(q < 2)
   {
      omp_set_num_threads(nnz/2);
      nthr = omp_get_max_threads();
      q = nnz/nthr;
      rest = nnz%nthr;
   }

   std::vector<T> result_array(nthr);

   #pragma omp parallel private(myid, first, last, psum)
   {
      myid = omp_get_thread_num();
      first = myid*q;

      if(myid == nthr-1)
      {
         last = (myid+1)*q+rest;
      }
      else
      {
         last = (myid+1)*q;
      }

      psum = values[first];
      for(size_t i = first+1; i < last; ++i)
      {
         psum = m_reduceFunc->CPU(psum, values[i]);
      }
      result_array[myid] = psum;
   }

   T tempResult = result_array.at(0);
   for(typename std::vector<T>::iterator it = ++result_array.begin(); it != result_array.end(); ++it)
   {
      tempResult = m_reduceFunc->CPU(tempResult, *it);
   }

   return tempResult;
}


/*!
 *  Performs the Reduction on non-zero elements of a SparseMatrix. Returns a \em SkePU vector of reduction result.
 *  Using \em OpenMP as backend. Can apply two different algorithms depending upon whether the workload is
 *  regular or irregular across different rows.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \param reducePolicy The policy specifying how reduction will be performed, can be either REDUCE_ROW_WISE_ONLY of REDUCE_COL_WISE_ONLY
 *  \return A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
skepu::Vector<T> Reduce<ReduceFunc, ReduceFunc>::OMP(SparseMatrix<T>& input, ReducePolicy reducePolicy)
{
   DEBUG_TEXT_LEVEL1("REDUCE SparseMatrix[] OpenMP\n")

   SparseMatrix<T> *matrix = NULL;

   if(reducePolicy==REDUCE_COL_WISE_ONLY)
      matrix = &(~input);
   else // assume  reducePolict==REDUCE_ROW_WISE_ONLY)
      matrix = &input;

   // Make sure we are properly synched with device data
   matrix->updateHost();

   size_t rows = matrix->total_rows();
   size_t nnz = matrix->total_nnz();

   omp_set_num_threads(m_execPlan->numOmpThreads(nnz));

   unsigned int numThreads = omp_get_max_threads();

   if(numThreads>rows) // when number of threads is less than processed rows, set 1 thread per row.
   {
      numThreads = rows;
      omp_set_num_threads(numThreads);
   }

   // Now, we safely assume that there are at least as number of rows to process as #threads available

   skepu::Vector<T> result(rows);

   T *res = result.getAddress();

   // two different kernels, depend upon std-dev, distribution of non-zero elements across rows is regular or ir-regular
   ompRegularWorkload<ReduceFunc, T>(m_reduceFunc, *matrix, res, numThreads);
//    ompIrregularWorkload<ReduceFunc, T>(m_reduceFunc, *matrix, res);

   return result;
}






/*!
 *  Performs the Reduction on a range of elements. Returns a scalar result. Divides the elements among all
 *  \em OpenMP threads and does reduction of the parts in parallel. The results from each thread are then
 *  reduced on the CPU.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type Reduce<ReduceFunc, ReduceFunc>::OMP(InputIterator inputBegin, InputIterator inputEnd)
{
   DEBUG_TEXT_LEVEL1("REDUCE OpenMP\n")

   omp_set_num_threads(m_execPlan->numOmpThreads(inputEnd-inputBegin));

   // Make sure we are properly synched with device data
   inputBegin.getParent().updateHost();

   size_t n = inputEnd-inputBegin;
   unsigned int nthr = omp_get_max_threads();
   size_t q = n/nthr;
   size_t rest = n%nthr;
   unsigned int myid;
   size_t first, last;
   typename InputIterator::value_type psum;

   if(q < 2)
   {
      omp_set_num_threads(n/2);
      nthr = omp_get_max_threads();
      q = n/nthr;
      rest = n%nthr;
   }

   std::vector<typename InputIterator::value_type> result_array(nthr);

   #pragma omp parallel private(myid, first, last, psum)
   {
      myid = omp_get_thread_num();
      first = myid*q;

      if(myid == nthr-1)
      {
         last = (myid+1)*q+rest;
      }
      else
      {
         last = (myid+1)*q;
      }

      psum = inputBegin(first);
      for(size_t i = first+1; i < last; ++i)
      {
         psum = m_reduceFunc->CPU(psum, inputBegin(i));
      }
      result_array[myid] = psum;
   }

   typename InputIterator::value_type tempResult = result_array.at(0);
   for(typename std::vector<typename InputIterator::value_type>::iterator it = ++result_array.begin(); it != result_array.end(); ++it)
   {
      tempResult = m_reduceFunc->CPU(tempResult, *it);
   }

   return tempResult;
}


}

#endif

