/*! \file reduce_common.h
 *  \brief Contains the definitions of common member functions for the Reduce skeleton that is used for both 1D and 2D reduction operations.
 */

#ifndef REDUCE_COMMON_HELPERS_H
#define REDUCE_COMMON_HELPERS_H


namespace skepu
{

#ifdef SKEPU_OPENMP

/*!
 * A function to do 1D reduction on OpenMP considering regular work-load per row. Useful for 2D dense matrix as well as structured sparse matrices.
 */
template <typename ReduceFunc, typename T>
void ompRegularWorkload(ReduceFunc *reduceFunc, SparseMatrix<T>& input, T *result_array, const unsigned int &numThreads)
{
   size_t rows = input.total_rows();
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
         typename SparseMatrix<T>::iterator it = input.begin(r);

         size_t size= it.size();
         if(size>0)
         {
            psum = it[0];

            for(size_t c=1; c<size; c++)
            {
               psum = reduceFunc->CPU(psum, it[c]);
            }
         }
         else
            psum = T();

         result_array[r] = psum;
      }
   }
}



/*!
 * A function to do 1D reduction on OpenMP considering ir-regular work-load per row. Useful for un-structured sparse matrices.
 */
template <typename ReduceFunc, typename T>
void ompIrregularWorkload(ReduceFunc *reduceFunc, SparseMatrix<T>& input, T *result_array)
{
   size_t rows = input.total_rows();

   T psum;

   // determine schedule at runtime, can be guided, dynamic with different chunk sizes
   #pragma omp parallel for private(psum) default(shared) schedule(runtime)
   for(size_t r= 0; r<rows; ++r)
   {
      typename SparseMatrix<T>::iterator it = input.begin(r);

      size_t size= it.size();
      if(size>0)
      {
         psum = it[0];

         for(size_t c=1; c<size; c++)
         {
            psum = reduceFunc->CPU(psum, it[c]);
         }
      }
      else
         psum = T();

      result_array[r] = psum;
   }
}


#endif




}

#endif
