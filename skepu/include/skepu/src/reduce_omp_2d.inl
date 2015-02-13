/*! \file reduce_omp_2d.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the 2DReduce skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>
#include <iostream>
#include <vector>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on a
 *  input Matrix. Returns a scalar result.
 *  Using the \em OpenMP as backend.
 *
 *  \param input A matrix on which the 2D reduction will be performed on.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::OMP(Matrix<T>& input)
{
   DEBUG_TEXT_LEVEL1("REDUCE 2D Matrix OpenMP\n")

   // Make sure we are properly synched with device data
   input.updateHost();

   size_t rows = input.total_rows();
   size_t cols = input.total_cols();

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

   std::vector<T> result_array(rows);

   T *data = input.getAddress();

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
            psum = m_reduceFuncRowWise->CPU(psum, data[base+c]);
         }
         result_array[r] = psum;
      }
   }



   T tempResult;

   if((rows/numThreads)>8) // if sufficient work to do it in parallel
      tempResult = ompVectorReduce(result_array, numThreads);
   else
   {
      // do it sequentially
      tempResult = result_array.at(0);
      for(typename std::vector<T>::iterator it = ++result_array.begin(); it != result_array.end(); ++it)
      {
         tempResult = m_reduceFuncColWise->CPU(tempResult, *it);
      }
   }

   return tempResult;

}

/*!
 *  A helper provate method used to do final 1D reduction. Used internally.
 *
 *  \param input A matrix on which the 2D reduction will be performed on.
 *  \param numThreads Number fo OpenMP threads to be used.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::ompVectorReduce(std::vector<T> &input, const size_t &numThreads)
{
   omp_set_num_threads(numThreads);

   size_t n = input.size();
   unsigned int nthr = omp_get_max_threads();
   size_t q = n/nthr;
   size_t rest = n%nthr;
   unsigned int myid;
   size_t first, last;
   T psum;

   if(q < 2)
   {
      omp_set_num_threads(n/2);
      nthr = omp_get_max_threads();
      q = n/nthr;
      rest = n%nthr;
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

      psum = input[first];
      for(size_t i = first+1; i < last; ++i)
      {
         psum = m_reduceFuncColWise->CPU(psum, input[i]);
      }
      result_array[myid] = psum;
   }

   T tempResult = result_array.at(0);
   for(typename std::vector<T>::iterator it = ++result_array.begin(); it != result_array.end(); ++it)
   {
      tempResult = m_reduceFuncColWise->CPU(tempResult, *it);
   }

   return tempResult;
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
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::OMP(SparseMatrix<T>& input)
{
   DEBUG_TEXT_LEVEL1("REDUCE 2D SparseMatrix OpenMP\n")

   // Make sure we are properly synched with device data
   input.updateHost();

   size_t rows = input.total_rows();
   size_t nnz = input.total_nnz();

   omp_set_num_threads(m_execPlan->numOmpThreads(nnz));

   unsigned int numThreads = omp_get_max_threads();

   if(numThreads>rows) // when number of threads is less than processed rows, set 1 thread per row.
   {
      numThreads = rows;
      omp_set_num_threads(numThreads);
   }

   // Now, we safely assume that there are at least as number of rows to process as #threads available

   std::vector<T> result_array(rows);

   // two different kernels, depend upon std-dev, distribution of non-zero elements across rows is regular or ir-regular
   ompRegularWorkload<ReduceFuncRowWise, T>(m_reduceFuncRowWise, input, &result_array[0], numThreads);
//    ompIrregularWorkload<ReduceFuncRowWise, T>(m_reduceFuncRowWise, input, &result_array[0]);

   // final reduction
   T tempResult;

   if((rows/numThreads)>8) // if sufficient work to do it in parallel
      tempResult = ompVectorReduce(result_array, numThreads);
   else
   {
      // do it sequentially
      tempResult = result_array.at(0);
      for(typename std::vector<T>::iterator it = ++result_array.begin(); it != result_array.end(); ++it)
      {
         tempResult = m_reduceFuncColWise->CPU(tempResult, *it);
      }
   }

   return tempResult;
}






}

#endif

