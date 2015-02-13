/*! \file map_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the Map skeleton.
 */

#include <iostream>

#include "debug.h"
#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs mapping on \em one vector on the \em CPU. Input is used as output. The function is a wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd).
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(Vector<T>& input)
{
   CPU(input.begin(), input.end());
}


/*!
 *  Performs mapping with \em one vector on the \em CPU. Seperate output is used. The function is a wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin).
 *
 *  \param input A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(Vector<T>& input, Vector<T>& output)
{
   CPU(input.begin(), input.end(), output.begin());
}



/*!
 *  Performs mapping on \em one matrix on the \em CPU. Input is used as output. The function is a wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd).
 *
 *  \param input A matrix which the mapping will be performed on. It will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(Matrix<T>& input)
{
   CPU(input.begin(), input.end());
}


/*!
 *  Performs mapping with \em one matrix on the \em CPU. Seperate output is used. The function is a wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin).
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(Matrix<T>& input, Matrix<T>& output)
{
   CPU(input.begin(), input.end(), output.begin());
}

/*!
 *  Performs the Map with \em one element range on the \em CPU. Input is used as output. The Map skeleton needs to
 *  be created with a \em unary user function. The computation is done by iterating over all the elements in the range
 *  and applying the user function on each of them.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 */
template <typename MapFunc>
template <typename InputIterator>
void Map<MapFunc>::CPU(InputIterator inputBegin, InputIterator inputEnd)
{
   DEBUG_TEXT_LEVEL1("MAP CPU\n")

   //Make sure we are properly synched with device data
   inputBegin.getParent().updateHostAndInvalidateDevice();

   // Uses operator() instead of * or [] to avoid implicit synchronization and
   // unnecessary function calls.
   for(; inputBegin != inputEnd; ++inputBegin)
   {
      inputBegin(0) = m_mapFunc->CPU(inputBegin(0));
   }
}

/*!
 *  Performs the Map with \em one element range on the \em CPU. Seperate output range is used. The Map skeleton needs to
 *  be created with a \em unary user function. The computation is done by iterating over all the elements in the range
 *  and applying the user function on each of them.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param outputBegin An iterator to the first element of the output range.
 */
template <typename MapFunc>
template <typename InputIterator, typename OutputIterator>
void Map<MapFunc>::CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin)
{
   DEBUG_TEXT_LEVEL1("MAP CPU\n")

   //Make sure we are properly synched with device data
   inputBegin.getParent().updateHost();
   outputBegin.getParent().invalidateDeviceData();

   // Uses operator() instead of * or [] to avoid implicit synchronization and
   // unnecessary function calls.
   for(; inputBegin != inputEnd; ++inputBegin, ++outputBegin)
   {
      outputBegin(0) = m_mapFunc->CPU(inputBegin(0));
   }
}

/*!
 *  Performs mapping with \em two vectors on the \em CPU. Seperate output is used. The function is a wrapper for
 *  CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(Vector<T>& input1, Vector<T>& input2, Vector<T>& output)
{
   CPU(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin());
}


/*!
 *  Performs mapping with \em two matrices on the \em CPU. Seperate output is used. The function is a wrapper for
 *  CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin).
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& output)
{
   CPU(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin());
}

/*!
 *  Performs the Map with \em two element ranges on the \em CPU. Seperate output range is used. The Map skeleton needs to
 *  be created with a \em binary user function. The computation is done by iterating over all the elements in the ranges
 *  and applying the user function on each of them.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param outputBegin An iterator to the first element of the output range.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
void Map<MapFunc>::CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin)
{
   DEBUG_TEXT_LEVEL1("MAP CPU\n")

   //Make sure we are properly synched with device data
   input1Begin.getParent().updateHost();
   input2Begin.getParent().updateHost();
   outputBegin.getParent().invalidateDeviceData();

   // Uses operator() instead of * or [] to avoid implicit synchronization and
   // unnecessary function calls.
   for(; input1Begin != input1End; ++input1Begin, ++input2Begin, ++outputBegin)
   {
      outputBegin(0) = m_mapFunc->CPU(input1Begin(0), input2Begin(0));
   }
}

/*!
 *  Performs mapping with \em three vectors on the \em CPU. Seperate output is used. The function is a wrapper for
 *  CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param input3 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, Vector<T>& output)
{
   CPU(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), output.begin());
}


/*!
 *  Performs mapping with \em three matrices on the \em CPU. Seperate output is used. The function is a wrapper for
 *  CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin).
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param input3 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, Matrix<T>& output)
{
   CPU(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), output.begin());
}


/*!
 *  Performs the Map with \em three vector element ranges on the \em CPU. Seperate output range is used. The Map skeleton needs to
 *  be created with a \em trinary user function. The computation is done by iterating over all the elements in the ranges
 *  and applying the user function on each of them.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param input3Begin An iterator to the first element in the third range.
 *  \param input3End An iterator to the last element of the third range.
 *  \param outputBegin An iterator to the first element of the output range.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
void Map<MapFunc>::CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin)
{
   DEBUG_TEXT_LEVEL1("MAP CPU\n")

   //Make sure we are properly synched with device data
   input1Begin.getParent().updateHost();
   input2Begin.getParent().updateHost();
   input3Begin.getParent().updateHost();
   outputBegin.getParent().invalidateDeviceData();

   // Uses operator() instead of * or [] to avoid implicit synchronization and
   // unnecessary function calls.
   for(; input1Begin != input1End; ++input1Begin, ++input2Begin, ++input3Begin, ++outputBegin)
   {
      outputBegin(0) = m_mapFunc->CPU(input1Begin(0), input2Begin(0), input3Begin(0));
   }
}

}

