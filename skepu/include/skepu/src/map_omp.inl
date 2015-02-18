/*! \file map_omp.inl
 *  \brief Contains the definitions of OpenMP specific member functions for the Map skeleton.
 */

#ifdef SKEPU_OPENMP

#include <omp.h>
#include <iostream>

#include "debug.h"
#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs mapping on \em one vector with \em OpenMP used as backend. Input is used as output. The function is a wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin).
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(Vector<T>& input)
{
   OMP(input.begin(), input.end(), input.begin());
}

/*!
 *  Performs mapping on \em one vector with \em OpenMP used as backend. Seperate output vector is used. The function is a wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin).
 *
 *  \param input A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(Vector<T>& input, Vector<T>& output)
{
   OMP(input.begin(), input.end(), output.begin());
}



/*!
 *  Performs mapping on \em one matrix with \em OpenMP used as backend. Input is used as output. The function is a wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin).
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(Matrix<T>& input)
{
   OMP(input.begin(), input.end(), input.begin());
}

/*!
 *  Performs mapping on \em one matrix with \em OpenMP used as backend. Seperate output matrix is used. The function is a wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin).
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(Matrix<T>& input, Matrix<T>& output)
{
   OMP(input.begin(), input.end(), output.begin());
}


/*!
 *  Performs the Map with \em one element range using the \p OpenMP backend. Input is used as output. The function is a wrapper for
 *  OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin).
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 */
template <typename MapFunc>
template <typename InputIterator>
void Map<MapFunc>::OMP(InputIterator inputBegin, InputIterator inputEnd)
{
   OMP(inputBegin, inputEnd, inputBegin);
}

/*!
 *  Performs the Map with \em one element range using the \p OpenMP backend. Seperate output range is used. The Map skeleton needs to
 *  be created with a \em unary user function. The computation is done by iterating over all the elements in the range
 *  and applying the user function on each of them. The range is automatically divided amongst threads by OpenMP.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param outputBegin An iterator to the first element of the output range.
 */
template <typename MapFunc>
template <typename InputIterator, typename OutputIterator>
void Map<MapFunc>::OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin)
{
   DEBUG_TEXT_LEVEL1("MAP OpenMP\n")

   size_t n = inputEnd - inputBegin;

   //Make sure we are properly synched with device data
   inputBegin.getParent().updateHost();
   outputBegin.getParent().invalidateDeviceData();

   omp_set_num_threads(m_execPlan->numOmpThreads(inputEnd-inputBegin));

   #pragma omp parallel for
   for(size_t i = 0; i < n; ++i)
   {
      outputBegin(i) = m_mapFunc->CPU(inputBegin(i));
   }
}

/*!
 *  Performs mapping on \em two vectors with \em OpenMP used as backend. Seperate output vector is used. The function is a wrapper for
 *  OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(Vector<T>& input1, Vector<T>& input2, Vector<T>& output)
{
   OMP(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin());
}


/*!
 *  Performs mapping on \em two matrices with \em OpenMP used as backend. Seperate output matrix is used. The function is a wrapper for
 *  OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin).
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& output)
{
   OMP(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin());
}



/*!
 *  Performs the Map with \em two element ranges using the \p OpenMP backend. Seperate output range is used. The Map skeleton needs to
 *  be created with a \em binary user function. The computation is done by iterating over all the elements in the ranges
 *  and applying the user function on each of them. The ranges is automatically divided amongst threads by OpenMP.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param outputBegin An iterator to the first element of the output range.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
void Map<MapFunc>::OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin)
{
   DEBUG_TEXT_LEVEL1("MAP OpenMP\n")

   size_t n = input1End - input1Begin;

   input1Begin.getParent().updateHost();
   input2Begin.getParent().updateHost();
   outputBegin.getParent().invalidateDeviceData();

   omp_set_num_threads(m_execPlan->numOmpThreads(input1End-input1Begin));

   #pragma omp parallel for
   for(size_t i = 0; i < n; ++i)
   {
      outputBegin(i) = m_mapFunc->CPU(input1Begin(i), input2Begin(i));
   }
}

/*!
 *  Performs mapping on \em three vectors with \em OpenMP used as backend. Seperate output vector is used. The function is a wrapper for
 *  OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param input3 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, Vector<T>& output)
{
   OMP(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), output.begin());
}


/*!
 *  Performs mapping on \em three matrices with \em OpenMP used as backend. Seperate output matrix is used. The function is a wrapper for
 *  OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin).
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param input3 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, Matrix<T>& output)
{
   OMP(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), output.begin());
}

/*!
 *  Performs the Map with \em three element ranges using the \p OpenMP backend. Seperate output range is used. The Map skeleton needs to
 *  be created with a \em trinary user function. The computation is done by iterating over all the elements in the ranges
 *  and applying the user function on each of them. The ranges is automatically divided amongst threads by OpenMP.
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
void Map<MapFunc>::OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin)
{
   DEBUG_TEXT_LEVEL1("MAP OpenMP\n")

   size_t n = input1End - input1Begin;

   //Make sure we are properly synched with device data
   input1Begin.getParent().updateHost();
   input2Begin.getParent().updateHost();
   input3Begin.getParent().updateHost();
   outputBegin.getParent().invalidateDeviceData();

   omp_set_num_threads(m_execPlan->numOmpThreads(input1End-input1Begin));

   #pragma omp parallel for
   for(size_t i = 0; i < n; ++i)
   {
      outputBegin(i) = m_mapFunc->CPU(input1Begin(i), input2Begin(i), input3Begin(i));
   }
}


}

#endif

