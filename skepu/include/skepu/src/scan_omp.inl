/*! \file scan_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the Scan skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>
#include <iostream>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs the Scan on a whole Vector with itself as output using \em OpenMP as backend. A wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd, ScanType type, typename InputIterator::value_type init).
 *
 *  \param input A vector which will be scanned. It will be overwritten with the result.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::OMP(Vector<T>& input, ScanType type, T init)
{
   OMP(input.begin(), input.end(), type, init);
}




/*!
 *  Performs the Scan on a whole Matrix with itself as output using \em OpenMP as backend.
 *
 *  \param input A matrix which will be scanned. It will be overwritten with the result.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::OMP(Matrix<T>& input, ScanType type, T init)
{
   DEBUG_TEXT_LEVEL1("SCAN OpenMP Matrix\n")

   // Make sure we are properly synched with device data
   input.updateHostAndInvalidateDevice();

   T *data = input.getAddress();

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

   if(type == INCLUSIVE)
   {
      #pragma omp parallel private(myid, firstRow, lastRow)
      {
         myid = omp_get_thread_num();

         firstRow = myid*rowsPerThread;

         if(myid!=0)
            firstRow += (myid<restRows)? myid:restRows;

         if(myid < restRows)
            lastRow = firstRow+rowsPerThread+1;
         else
            lastRow = firstRow+rowsPerThread;

         T *local = data + firstRow*cols;

         for(size_t r= firstRow; r<lastRow; ++r)
         {
            for(size_t c=1; c<cols; ++c)
            {
               local[c] = m_scanFunc->CPU(local[c-1], local[c]);
            }
            local += cols;
         }
      }
   }
   else
   {
      #pragma omp parallel private(myid, firstRow, lastRow)
      {
         myid = omp_get_thread_num();

         firstRow = myid*rowsPerThread;

         if(myid!=0)
            firstRow += (myid<restRows)? myid:restRows;

         if(myid < restRows)
            lastRow = firstRow+rowsPerThread+1;
         else
            lastRow = firstRow+rowsPerThread;

         T *local = data + firstRow*cols;

         for(size_t r= firstRow; r<lastRow; ++r)
         {
            T out_before = init;
            T out_current = local[0];

            local[0] = out_before;
            for(size_t c=1; c<cols; ++c)
            {
               out_before = local[c];
               local[c] = m_scanFunc->CPU(out_current, local[c-1]);
               T temp = out_before;
               out_before = out_current;
               out_current = temp;
            }
            local += cols;
         }
      }
   }
}




/*!
 *  Performs the Scan on a range of elements with the same range as output using \em OpenMP as backend.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename InputIterator>
void Scan<ScanFunc>::OMP(InputIterator inputBegin, InputIterator inputEnd, ScanType type, typename InputIterator::value_type init)
{
   DEBUG_TEXT_LEVEL1("SCAN OpenMP\n")

   omp_set_num_threads(m_execPlan->numOmpThreads(inputEnd-inputBegin));

   //Make sure we are properly synched with device data
   inputBegin.getParent().updateHostAndInvalidateDevice();

   // Setup parameters needed to parallelize with OpenMP
   size_t n = inputEnd-inputBegin;
   unsigned int nthr = omp_get_max_threads();
   size_t q = n/nthr;
   size_t rest = n%nthr;
   unsigned int myid;
   size_t first, last;

   if(q < 2)
   {
      omp_set_num_threads(n/2);
      nthr = omp_get_max_threads();
      q = n/nthr;
      rest = n%nthr;
   }

   // Array to store partial thread results in.
   std::vector<typename InputIterator::value_type> offset_array(nthr);

   if(type == INCLUSIVE)
   {
      #pragma omp parallel private(myid, first, last)
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

         // First let each thread make their own scan and saved the result in a partial result array.
         for(size_t i = first+1; i < last; ++i)
         {
            inputBegin(i) = m_scanFunc->CPU(inputBegin(i-1), inputBegin(i));
         }
         offset_array[myid] = inputBegin(last-1);

         // Let the master thread scan the partial result array
         #pragma omp barrier
         #pragma omp master
         {
            for(unsigned int i = 1; i < nthr; ++i)
            {
               offset_array[i] = m_scanFunc->CPU(offset_array[i-1], offset_array[i]);
            }
         }

         #pragma omp barrier
         if(myid != 0)
         {
            // Add the scanned partial results to each threads work batch.
            for(size_t i = first; i < last; ++i)
            {
               inputBegin(i) = m_scanFunc->CPU(inputBegin(i), offset_array[myid-1]);
            }
         }

      }
   }
   else
   {
      #pragma omp parallel private(myid, first, last)
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

         // First let each thread make their own scan and saved the result in a partial result array.
         typename InputIterator::value_type out_before = init;
         typename InputIterator::value_type out_current = inputBegin(first);
         typename InputIterator::value_type psum = inputBegin(first);
         inputBegin(first) = out_before;
         for(size_t i = first+1; i < last; ++i)
         {
            psum = m_scanFunc->CPU(psum, inputBegin(i));
            out_before = inputBegin(i);
            inputBegin(i) = m_scanFunc->CPU(out_current, inputBegin(i-1));
            typename InputIterator::value_type temp = out_before;
            out_before = out_current;
            out_current = temp;
         }
         offset_array[myid] = psum;

         // Let the master thread scan the partial result array
         #pragma omp barrier
         #pragma omp master
         {
            for(unsigned int i = 1; i < nthr; ++i)
            {
               offset_array[i] = m_scanFunc->CPU(offset_array[i-1], offset_array[i]);
            }
         }

         #pragma omp barrier
         if(myid != 0)
         {
            // Add the scanned partial results to each threads work batch.
            for(size_t i = first; i < last; ++i)
            {
               inputBegin(i) = m_scanFunc->CPU(inputBegin(i), offset_array[myid-1]);
            }
         }

      }
   }
}

/*!
 *  Performs the Scan on a whole Vector with a seperate output vector using \em OpenMP as backend. Wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init).
 *
 *  \param input A vector which will be scanned.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::OMP(Vector<T>& input, Vector<T>& output, ScanType type, T init)
{
   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.size());
   }

   OMP(input.begin(), input.end(), output.begin(), type, init);
}



/*!
 *  Performs the Scan on a whole Matrix with a separate Matrix as output using \em OpenMP as backend.
 *
 *  \param input A matrix which will be scanned.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::OMP(Matrix<T>& input, Matrix<T>& output, ScanType type, T init)
{
   DEBUG_TEXT_LEVEL1("SCAN OpenMP Matrix\n")

   // Make sure we are properly synched with device data
   input.updateHost();
   output.invalidateDeviceData();

   if(input.total_rows() != output.total_rows() || input.total_cols() != output.total_cols())
   {
      output.clear();
      output.resize(input.total_rows(), input.total_cols());
   }

   T *in_data = input.getAddress();
   T *out_data = output.getAddress();

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

   if(type == INCLUSIVE)
   {
      #pragma omp parallel private(myid, firstRow, lastRow)
      {
         myid = omp_get_thread_num();

         firstRow = myid*rowsPerThread;

         if(myid!=0)
            firstRow += (myid<restRows)? myid:restRows;

         if(myid < restRows)
            lastRow = firstRow+rowsPerThread+1;
         else
            lastRow = firstRow+rowsPerThread;

         T *in_local = in_data + firstRow*cols;
         T *out_local = out_data + firstRow*cols;

         for(size_t r= firstRow; r<lastRow; ++r)
         {
            out_local[0] = in_local[0];
            for(size_t c=1; c<cols; ++c)
            {
               out_local[c] = m_scanFunc->CPU(out_local[c-1], in_local[c]);
            }
            in_local += cols;
            out_local += cols;
         }
      }
   }
   else
   {
      #pragma omp parallel private(myid, firstRow, lastRow)
      {
         myid = omp_get_thread_num();

         firstRow = myid*rowsPerThread;

         if(myid!=0)
            firstRow += (myid<restRows)? myid:restRows;

         if(myid < restRows)
            lastRow = firstRow+rowsPerThread+1;
         else
            lastRow = firstRow+rowsPerThread;

         T *in_local = in_data + firstRow*cols;
         T *out_local = out_data + firstRow*cols;

         for(size_t r= firstRow; r<lastRow; ++r)
         {
            out_local[0] = init;
            for(size_t c=1; c<cols; ++c)
            {
               out_local[c] = m_scanFunc->CPU(out_local[c-1], in_local[c-1]);
            }
            in_local += cols;
            out_local += cols;
         }
      }
   }
}

/*!
 *  Performs the Scan on a range of elements with a seperate output range using \em OpenMP as backend.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename InputIterator, typename OutputIterator>
void Scan<ScanFunc>::OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init)
{
   DEBUG_TEXT_LEVEL1("SCAN OpenMP\n")

   omp_set_num_threads(m_execPlan->numOmpThreads(inputEnd-inputBegin));

   //Make sure we are properly synched with device data
   outputBegin.getParent().invalidateDeviceData();
   inputBegin.getParent().updateHost();

   // Setup parameters needed to parallelize with OpenMP
   size_t n = inputEnd-inputBegin;
   unsigned int nthr = omp_get_max_threads();
   size_t q = n/nthr;
   size_t rest = n%nthr;
   unsigned int myid;
   size_t first, last;

   if(q < 2)
   {
      omp_set_num_threads(n/2);
      nthr = omp_get_max_threads();
      q = n/nthr;
      rest = n%nthr;
   }

   // Array to store partial thread results in.
   std::vector<typename InputIterator::value_type> offset_array(nthr);

   if(type == INCLUSIVE)
   {
      #pragma omp parallel private(myid, first, last)
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

         // First let each thread make their own scan and saved the result in a partial result array.
         outputBegin(first) = inputBegin(first);
         for(size_t i = first+1; i < last; ++i)
         {
            outputBegin(i) = m_scanFunc->CPU(outputBegin(i-1), inputBegin(i));
         }
         offset_array[myid] = outputBegin(last-1);

         // Let the master thread scan the partial result array
         #pragma omp barrier
         #pragma omp master
         {
            for(unsigned int i = 1; i < nthr; ++i)
            {
               offset_array[i] = m_scanFunc->CPU(offset_array[i-1], offset_array[i]);
            }
         }

         #pragma omp barrier
         if(myid != 0)
         {
            // Add the scanned partial results to each threads work batch.
            for(size_t i = first; i < last; ++i)
            {
               outputBegin(i) = m_scanFunc->CPU(outputBegin(i), offset_array[myid-1]);
            }
         }

      }
   }
   else
   {
      #pragma omp parallel private(myid, first, last)
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

         // First let each thread make their own scan and saved the result in a partial result array.
         typename InputIterator::value_type psum = inputBegin(first);
         outputBegin(first) = init;
         for(size_t i = first+1; i < last; ++i)
         {
            outputBegin(i) = m_scanFunc->CPU(outputBegin(i-1), inputBegin(i-1));
            psum = m_scanFunc->CPU(psum, inputBegin(i));
         }
         offset_array[myid] = psum;

         // Let the master thread scan the partial result array
         #pragma omp barrier
         #pragma omp master
         {
            for(unsigned int i = 1; i < nthr; ++i)
            {
               offset_array[i] = m_scanFunc->CPU(offset_array[i-1], offset_array[i]);
            }
         }

         #pragma omp barrier
         if(myid != 0)
         {
            // Add the scanned partial results to each threads work batch.
            for(size_t i = first; i < last; ++i)
            {
               outputBegin(i) = m_scanFunc->CPU(outputBegin(i), offset_array[myid-1]);
            }
         }

      }
   }
}

}

#endif

