/*! \file generate_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the Generate skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>
#include <iostream>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output vector which is resized to numElements. A wrapper for
 *  OMP(size_t numElements, OutputIterator outputBegin). For the \em OpenMP backend.
 *
 *  \param numElements The number of elements to be generated.
 *  \param output The output vector which will be overwritten with the generated values.
 */
template <typename GenerateFunc>
template <typename T>
void Generate<GenerateFunc>::OMP(size_t numElements, Vector<T>& output)
{
   if(output.size() != numElements)
   {
      output.clear();
      output.resize(numElements);
   }

   OMP(numElements, output.begin());
}



/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output matrix which is resized to numElements. For the \em OpenMP backend.
 *
 *  \param numRows The number of rows to be generated.
 *  \param numCols The number of columns to be generated.
 *  \param output The output matrix which will be overwritten with the generated values.
 */
template <typename GenerateFunc>
template <typename T>
void Generate<GenerateFunc>::OMP(size_t numRows, size_t numCols, Matrix<T>& output)
{
   DEBUG_TEXT_LEVEL1("GENERATE OpenMP Matrix\n")

   if((output.total_rows() != numRows) && (output.total_cols() != numCols))
   {
      output.clear();
      output.resize(numRows, numCols);
   }

   omp_set_num_threads(m_execPlan->numOmpThreads(numRows*numCols));

   //Make sure we are properly synched with device data
   output.invalidateDeviceData();

   // Uses operator() instead of [] or (,) to avoid implicit synchronization and
   // unnecessary function calls.
   #pragma omp parallel for
   for(size_t r = 0; r < numRows; ++r)
      for(size_t c = 0; c < numCols; ++c)
      {
         output(r*numCols+c) = m_generateFunc->CPU(c, r);
      }
}

/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output range. For the \em OpenMP backend.
 *
 *  \param numElements The number of elements to be generated.
 *  \param outputBegin An iterator pointing to the first element in the range which will be overwritten with generated values.
 */
template <typename GenerateFunc>
template <typename OutputIterator>
void Generate<GenerateFunc>::OMP(size_t numElements, OutputIterator outputBegin)
{
   DEBUG_TEXT_LEVEL1("GENERATE OpenMP\n")

   omp_set_num_threads(m_execPlan->numOmpThreads(numElements));

   //Make sure we are properly synched with device data
   outputBegin.getParent().invalidateDeviceData();

   // Uses operator() instead of * or [] to avoid implicit synchronization and
   // unnecessary function calls.
   #pragma omp parallel for
   for(size_t i = 0; i < numElements; ++i)
   {
      outputBegin(i) = m_generateFunc->CPU(i);
   }
}

}

#endif

