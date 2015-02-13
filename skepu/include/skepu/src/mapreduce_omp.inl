/*! \file mapreduce_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the MapReduce skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>
#include <iostream>
#include <vector>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs the Map on \em one Vector and Reduce on the result. Returns a scalar result. A wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd).
 *  Using the \em OpenMP as backend.
 *
 *  \param input A vector which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::OMP(Vector<T>& input)
{
   return OMP(input.begin(), input.end());
}

/*!
 *  Performs the Map on \em one Matrix and Reduce on the result. Returns a scalar result. A wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd).
 *  Using the \em OpenMP as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::OMP(Matrix<T>& input)
{
   return OMP(input.begin(), input.end());
}

/*!
 *  Performs the Map on \em one range of elements and Reduce on the result. Returns a scalar result.
 *  Divides the elements among all \em OpenMP threads and does mapping and reduction of the parts in parallel.
 *  The results from each thread are then reduced on the CPU.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type MapReduce<MapFunc, ReduceFunc>::OMP(InputIterator inputBegin, InputIterator inputEnd)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE OpenMP\n")

   omp_set_num_threads(m_execPlan->numOmpThreads(inputEnd-inputBegin));

   //Make sure we are properly synched with device data
   inputBegin.getParent().updateHost();

   size_t n = inputEnd-inputBegin;
   unsigned int nthr = omp_get_max_threads();
   size_t q = n/nthr;
   size_t rest = n%nthr;
   unsigned int myid;
   size_t first, last;
   typename InputIterator::value_type psum;
   typename InputIterator::value_type tempMap;

   if(q < 2)
   {
      omp_set_num_threads(n/2);
      nthr = omp_get_max_threads();
      q = n/nthr;
      rest = n%nthr;
   }

   std::vector<typename InputIterator::value_type> result_array(nthr);

   #pragma omp parallel private(myid, first, last, psum, tempMap)
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

      tempMap = m_mapFunc->CPU(inputBegin(first));
      psum = tempMap;
      for(size_t i = first+1; i < last; ++i)
      {
         tempMap = m_mapFunc->CPU(inputBegin(i));
         psum = m_reduceFunc->CPU(psum, tempMap);
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

/*!
 *  Performs the Map on \em two Vectors and Reduce on the result. Returns a scalar result. A wrapper for
 *  OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End).
 *  Using the \em OpenMP as backend.
 *
 *  \param input1 A Vector which the map and reduce will be performed on.
 *  \param input2 A Vector which the map and reduce will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::OMP(Vector<T>& input1, Vector<T>& input2)
{
   return OMP(input1.begin(), input1.end(), input2.begin(), input2.end());
}

/*!
 *  Performs the Map on \em two Matrices and Reduce on the result. Returns a scalar result. A wrapper for
 *  OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End).
 *  Using the \em OpenMP as backend.
 *
 *  \param input1 A Matrix which the map and reduce will be performed on.
 *  \param input2 A Matrix which the map and reduce will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::OMP(Matrix<T>& input1, Matrix<T>& input2)
{
   return OMP(input1.begin(), input1.end(), input2.begin(), input2.end());
}

/*!
 *  Performs the Map on \em two ranges of elements and Reduce on the result. Returns a scalar result.
 *  Divides the elements among all \em OpenMP threads and does mapping and reduction of the parts in parallel.
 *  The results from each thread are then reduced on the CPU.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE OpenMP\n")

   omp_set_num_threads(m_execPlan->numOmpThreads(input1End-input1Begin));

   size_t n = input1End - input1Begin;

   //Make sure we are properly synched with device data
   input1Begin.getParent().updateHost();
   input2Begin.getParent().updateHost();

   unsigned int nthr = omp_get_max_threads();
   size_t q = n/nthr;
   size_t rest = n%nthr;
   unsigned int myid;
   size_t first, last;
   typename Input1Iterator::value_type psum;
   typename Input1Iterator::value_type tempMap;

   if(q < 2)
   {
      omp_set_num_threads(n/2);
      nthr = omp_get_max_threads();
      q = n/nthr;
      rest = n%nthr;
   }

   std::vector<typename Input1Iterator::value_type> result_array(nthr);

   #pragma omp parallel private(myid, first, last, psum, tempMap)
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

      tempMap = m_mapFunc->CPU(input1Begin(first), input2Begin(first));
      psum = tempMap;
      for(size_t i = first+1; i < last; ++i)
      {
         tempMap = m_mapFunc->CPU(input1Begin(i), input2Begin(i));
         psum = m_reduceFunc->CPU(psum, tempMap);
      }
      result_array[myid] = psum;
   }

   typename Input1Iterator::value_type tempResult = result_array.at(0);
   for(typename std::vector<typename Input1Iterator::value_type>::iterator it = ++result_array.begin(); it != result_array.end(); ++it)
   {
      tempResult = m_reduceFunc->CPU(tempResult, *it);
   }

   return tempResult;
}

/*!
 *  Performs the Map on \em three Vectors and Reduce on the result. Returns a scalar result. A wrapper for
 *  OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End).
 *  Using the \em OpenMP as backend.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param input3 A vector which the mapping will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::OMP(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3)
{
   return OMP(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end());
}

/*!
 *  Performs the Map on \em three Matrices and Reduce on the result. Returns a scalar result. A wrapper for
 *  OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End).
 *  Using the \em OpenMP as backend.
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param input3 A matrix which the mapping will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::OMP(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3)
{
   return OMP(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end());
}


/*!
 *  Performs the Map on \em three ranges of elements and Reduce on the result. Returns a scalar result.
 *  Divides the elements among all \em OpenMP threads and does mapping and reduction of the parts in parallel.
 *  The results from each thread are then reduced on the CPU.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param input3Begin An iterator to the first element in the third range.
 *  \param input3End An iterator to the last element of the third range.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE OpenMP\n")

   omp_set_num_threads(m_execPlan->numOmpThreads(input1End-input1Begin));

   size_t n = input1End - input1Begin;

   //Make sure we are properly synched with device data
   input1Begin.getParent().updateHost();
   input2Begin.getParent().updateHost();
   input3Begin.getParent().updateHost();

   unsigned int nthr = omp_get_max_threads();
   size_t q = n/nthr;
   size_t rest = n%nthr;
   unsigned int myid;
   size_t first, last;
   typename Input1Iterator::value_type psum;
   typename Input1Iterator::value_type tempMap;

   if(q < 2)
   {
      omp_set_num_threads(n/2);
      nthr = omp_get_max_threads();
      q = n/nthr;
      rest = n%nthr;
   }

   std::vector<typename Input1Iterator::value_type> result_array(nthr);

   #pragma omp parallel private(myid, first, last, psum, tempMap)
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

      tempMap = m_mapFunc->CPU(input1Begin(first), input2Begin(first), input3Begin(first));
      psum = tempMap;
      for(size_t i = first+1; i < last; ++i)
      {
         tempMap = m_mapFunc->CPU(input1Begin(i), input2Begin(i), input3Begin(i));
         psum = m_reduceFunc->CPU(psum, tempMap);
      }
      result_array[myid] = psum;
   }

   typename Input1Iterator::value_type tempResult = result_array.at(0);
   for(typename std::vector<typename Input1Iterator::value_type>::iterator it = ++result_array.begin(); it != result_array.end(); ++it)
   {
      tempResult = m_reduceFunc->CPU(tempResult, *it);
   }

   return tempResult;
}


}

#endif

