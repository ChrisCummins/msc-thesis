/*! \file mapreduce_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapReduce skeleton.
 */

#include <iostream>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs the Map on \em one Vector and Reduce on the result. Returns a scalar result. A wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd).
 *  Using the \em CPU as backend.
 *
 *  \param input A vector which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CPU(Vector<T>& input)
{
   return CPU(input.begin(), input.end());
}


/*!
 *  Performs the Map on \em one Matrix and Reduce on the result. Returns a scalar result. A wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd).
 *  Using the \em CPU as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CPU(Matrix<T>& input)
{
   return CPU(input.begin(), input.end());
}

/*!
 *  Performs the Map on \em one range of elements and Reduce on the result. Returns a scalar result.
 *  Does the reduction on the \em CPU by iterating over all elements in the range.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type MapReduce<MapFunc, ReduceFunc>::CPU(InputIterator inputBegin, InputIterator inputEnd)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE CPU\n")

   //Make sure we are properly synched with device data
   inputBegin.getParent().updateHost();

   typename InputIterator::value_type tempMap = m_mapFunc->CPU(inputBegin(0));
   inputBegin++;
   typename InputIterator::value_type tempReduce = tempMap;
   for(; inputBegin != inputEnd; ++inputBegin)
   {
      tempMap = m_mapFunc->CPU(inputBegin(0));
      tempReduce = m_reduceFunc->CPU(tempReduce, tempMap);
   }

   return tempReduce;
}

/*!
 *  Performs the Map on \em two Vectors and Reduce on the result. Returns a scalar result. A wrapper for
 *  CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End).
 *  Using the \em CPU as backend.
 *
 *  \param input1 A Vector which the map and reduce will be performed on.
 *  \param input2 A Vector which the map and reduce will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CPU(Vector<T>& input1, Vector<T>& input2)
{
   return CPU(input1.begin(), input1.end(), input2.begin(), input2.end());
}


/*!
 *  Performs the Map on \em two matrices and Reduce on the result. Returns a scalar result. A wrapper for
 *  CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End).
 *  Using the \em CPU as backend.
 *
 *  \param input1 A matrix which the map and reduce will be performed on.
 *  \param input2 A matrix which the map and reduce will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CPU(Matrix<T>& input1, Matrix<T>& input2)
{
   return CPU(input1.begin(), input1.end(), input2.begin(), input2.end());
}

/*!
 *  Performs the Map on \em two ranges of elements and Reduce on the result. Returns a scalar result.
 *  Does the reduction on the \em CPU by iterating over all elements in the range.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE CPU\n")

   //Make sure we are properly synched with device data
   input1Begin.getParent().updateHost();
   input2Begin.getParent().updateHost();

   typename Input1Iterator::value_type tempMap = m_mapFunc->CPU(input1Begin(0), input2Begin(0));
   input1Begin++;
   input2Begin++;
   typename Input1Iterator::value_type tempReduce = tempMap;
   for(; input1Begin != input1End; ++input1Begin, ++input2Begin)
   {
      tempMap = m_mapFunc->CPU(input1Begin(0), input2Begin(0));
      tempReduce = m_reduceFunc->CPU(tempReduce, tempMap);
   }

   return tempReduce;
}

/*!
 *  Performs the Map on \em three Vectors and Reduce on the result. Returns a scalar result. A wrapper for
 *  CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End).
 *  Using the \em CPU as backend.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param input3 A vector which the mapping will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CPU(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3)
{
   return CPU(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end());
}

/*!
 *  Performs the Map on \em three matrices and Reduce on the result. Returns a scalar result. A wrapper for
 *  CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End).
 *  Using the \em CPU as backend.
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param input3 A matrix which the mapping will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CPU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3)
{
   return CPU(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end());
}


/*!
 *  Performs the Map on \em three ranges of elements and Reduce on the result. Returns a scalar result.
 *  Does the reduction on the \em CPU by iterating over all elements in the ranges.
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
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE CPU\n")

   //Make sure we are properly synched with device data
   input1Begin.getParent().updateHost();
   input2Begin.getParent().updateHost();
   input3Begin.getParent().updateHost();

   typename Input1Iterator::value_type tempMap = m_mapFunc->CPU(input1Begin(0), input2Begin(0), input3Begin(0));
   input1Begin++;
   input2Begin++;
   input3Begin++;
   typename Input1Iterator::value_type tempReduce = tempMap;
   for(; input1Begin != input1End; ++input1Begin, ++input2Begin, ++input3Begin)
   {
      tempMap = m_mapFunc->CPU(input1Begin(0), input2Begin(0), input3Begin(0));
      tempReduce = m_reduceFunc->CPU(tempReduce, tempMap);
   }

   return tempReduce;
}

}

