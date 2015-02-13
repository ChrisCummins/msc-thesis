/*! \file mapoverlap_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapOverlap skeleton.
 */

#include <iostream>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs the MapOverlap on a whole Vector on the \em CPU with itself as output. A wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad).
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CPU(Vector<T>& input, EdgePolicy poly, T pad)
{
   CPU(input.begin(), input.end(), poly, pad);
}

/*!
 *  Performs the MapOverlap on a range of elements on the \em CPU with the same range as output. Since a seperate output is needed, a
 *  temporary output vector is created and copied to the input vector at the end. This is rather inefficient and the
 *  two functions using a seperated output explicitly should be used instead.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename InputIterator>
void MapOverlap<MapOverlapFunc>::CPU(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad)
{
   Vector<typename InputIterator::value_type> output;
   output.clear();
   output.resize(inputEnd-inputBegin);

   CPU(inputBegin, inputEnd, output.begin(), poly, pad);

   for(InputIterator it = output.begin(); inputBegin != inputEnd; ++inputBegin, ++it)
   {
      *inputBegin = *it;
   }
}

/*!
 *  Performs the MapOverlap on a Matrix on the \em CPU with the same Matrix as output.
 *  Wrapper for  CPU(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy,
 *  EdgePolicy poly, typename InputIterator::value_type pad).
 *
 *  \param input A Matrix that is used for both input and output.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CPU(Matrix<T>& input, OverlapPolicy overlapPolicy, EdgePolicy poly, T pad)
{
   CPU(input.begin(), input.end(), overlapPolicy, poly, pad);
}

/*!
 *  Performs the MapOverlap on a range of matrix elements on the \em CPU with the same range as output. Since a seperate output is needed, a
 *  temporary output matrix is created and copied to the input matrix at the end. This is rather inefficient and the
 *  two functions using a seperated output explicitly should be used instead.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename InputIterator>
void MapOverlap<MapOverlapFunc>::CPU(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad)
{
   Matrix<typename InputIterator::value_type> output;
   output.clear();
   output.resize(inputBegin.getParent().total_rows(), inputBegin.getParent().total_cols());

   CPU(inputBegin, inputEnd, output.begin(), overlapPolicy, poly, pad);

   for(InputIterator it = output.begin(); inputBegin != inputEnd; ++inputBegin, ++it)
   {
      *inputBegin = *it;
   }
}


/*!
 *  Performs the MapOverlap on a whole Vector on the \em CPU with a seperate output vector. Wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad).
 *
 *  \param input A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CPU(Vector<T>& input, Vector<T>& output, EdgePolicy poly, T pad)
{
   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.size());
   }

   CPU(input.begin(), input.end(), output.begin(), poly, pad);
}

/*!
 *  Performs the MapOverlap on a range of elements on the \em CPU with a seperate output range.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad)
{
   DEBUG_TEXT_LEVEL1("MAPOVERLAP CPU\n")

   //Make sure we are properly synched with device data
   outputBegin.getParent().invalidateDeviceData();
   inputBegin.getParent().updateHost();

   size_t overlap = m_mapOverlapFunc->overlap;

   //Make two arrays, start and end with zero padding
   typename InputIterator::pointer startArray = new typename InputIterator::value_type[3*overlap];
   typename InputIterator::pointer endArray = new typename InputIterator::value_type[3*overlap];

   //Pad with zeros
   for(size_t i = 0; i < overlap; ++i)
   {
      if(poly == CONSTANT)
      {
         startArray[i] = pad;
         endArray[3*overlap-i-1] = pad;
      }
      else if(poly == CYCLIC)
      {
         startArray[i] = inputEnd(i-overlap);
         endArray[3*overlap-i-1] = inputBegin(overlap-i-1);
      }
      else if(poly == DUPLICATE)
      {
         startArray[i] = inputBegin(0);
         endArray[3*overlap-i-1] = inputEnd(-1);
      }
   }

   //Copy values
   for(size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
   {
      startArray[i] = inputBegin(j);
   }
   InputIterator it = inputEnd-2*overlap;
   for(size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
   {
      endArray[i] = it(j);
   }

   size_t n = inputEnd - inputBegin;
   for(size_t i = 0; i < n; ++i)
   {
      if(i < overlap)
      {
         outputBegin(i) = m_mapOverlapFunc->CPU(&startArray[i+overlap]);
      }
      else if(i+overlap >= n)
      {
         outputBegin(i) = m_mapOverlapFunc->CPU(&endArray[overlap+i-(n-overlap)]);
      }
      else
      {
         outputBegin(i) = m_mapOverlapFunc->CPU(&(inputBegin(i)));
      }
   }

   delete[] startArray;
   delete[] endArray;
}



/*!
 *  Performs the MapOverlap on a whole matrix on the \em CPU with a seperate output matrix. Wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad).
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CPU(Matrix<T>& input, Matrix<T>& output, OverlapPolicy overlapPolicy, EdgePolicy poly, T pad)
{
   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.total_rows(), input.total_cols());
   }

   CPU(input.begin(), input.end(), output.begin(), overlapPolicy, poly, pad);
}

/*!
 *  Performs the MapOverlap on a range of matrix elements on the \em CPU with a seperate output range.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad)
{
   DEBUG_TEXT_LEVEL1("MAPOVERLAP CPU\n")

   //Make sure we are properly synched with device data
   outputBegin.getParent().invalidateDeviceData();
   inputBegin.getParent().updateHost();

   if(overlapPolicy == OVERLAP_ROW_COL_WISE)
   {
      Matrix<typename InputIterator::value_type> tmp_m(outputBegin.getParent().total_rows(), outputBegin.getParent().total_cols());
      CPU_ROWWISE(inputBegin,inputEnd,tmp_m.begin(), poly, pad);
      CPU_COLWISE(tmp_m.begin(),tmp_m.end(),outputBegin, poly, pad);
   }
   else if(overlapPolicy == OVERLAP_COL_WISE)
      CPU_COLWISE(inputBegin,inputEnd,outputBegin, poly, pad);
   else
      CPU_ROWWISE(inputBegin,inputEnd,outputBegin, poly, pad);
}



/*!
 *  Performs the row-wise MapOverlap on a range of elements on the \em CPU with a seperate output range.
 *  Used internally by other methods to apply row-wise mapoverlap operation.
 *
 *  \param inputBegin A vector::iterator to the first element in the range.
 *  \param inputEnd A vector::iterator to the last element of the range.
 *  \param outputBegin A vector::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::CPU_ROWWISE(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad)
{
   size_t n= inputEnd - inputBegin;

   size_t rowWidth= inputBegin.getParent().total_cols();

   size_t overlap = m_mapOverlapFunc->overlap;
   m_mapOverlapFunc->setStride(1);

   //Make two arrays, start and end with zero padding
   typename InputIterator::pointer startArray = new typename InputIterator::value_type[3*overlap];
   typename InputIterator::pointer endArray = new typename InputIterator::value_type[3*overlap];

   for(size_t row=0; row< (n/rowWidth); row++)
   {
      inputEnd = inputBegin+rowWidth;

      //Pad with zeros
      for(size_t i = 0; i < overlap; ++i)
      {
         if(poly == CONSTANT)
         {
            startArray[i] = pad;
            endArray[3*overlap-i-1] = pad;
         }
         else if(poly == CYCLIC)
         {
            startArray[i] = inputEnd(i-overlap);
            endArray[3*overlap-i-1] = inputBegin(overlap-i-1);
         }
         else if(poly == DUPLICATE)
         {
            startArray[i] = inputBegin(0);
            endArray[3*overlap-i-1] = inputEnd(-1);
         }
      }

      //Copy values
      for(size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
      {
         startArray[i] = inputBegin(j);
      }
      InputIterator it = inputEnd-2*overlap;
      for(size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
      {
         endArray[i] = it(j);
      }

      size_t n = inputEnd - inputBegin;
      for(size_t i = 0; i < n; ++i)
      {
         if(i < overlap)
         {
            outputBegin(i) = m_mapOverlapFunc->CPU(&startArray[i+overlap]);
         }
         else if(i+overlap >= n)
         {
            outputBegin(i) = m_mapOverlapFunc->CPU(&endArray[overlap+i-(n-overlap)]);
         }
         else
         {
            outputBegin(i) = m_mapOverlapFunc->CPU(&(inputBegin(i)));
         }
      }
      inputBegin += rowWidth;
      outputBegin += rowWidth;
   }

   delete[] startArray;
   delete[] endArray;
}


/*!
 *  Performs the column-wise MapOverlap on a range of elements on the \em CPU with a seperate output range.
 *  Used internally by other methods to apply row-wise mapoverlap operation.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::CPU_COLWISE(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad)
{
   size_t n= inputEnd - inputBegin;

   size_t rowWidth= inputBegin.getParent().total_cols();
   size_t colWidth= inputBegin.getParent().total_rows();

   size_t overlap = m_mapOverlapFunc->overlap;

   //Make two arrays, start and end with zero padding
   typename InputIterator::pointer startArray = new typename InputIterator::value_type[3*overlap];
   typename InputIterator::pointer endArray = new typename InputIterator::value_type[3*overlap];

   size_t stride = rowWidth;

   for(size_t col=0; col< (n/colWidth); col++)
   {
      inputEnd = inputBegin + (rowWidth*(colWidth-1));

      //Pad with zeros
      for(size_t i = 0; i < overlap; ++i)
      {
         if(poly == CONSTANT)
         {
            startArray[i] = pad;
            endArray[3*overlap-i-1] = pad;
         }
         else if(poly == CYCLIC)
         {
            startArray[i] = inputEnd((i+1-overlap)*stride);
            endArray[3*overlap-i-1] = inputBegin((overlap-i-1)*stride);
         }
         else if(poly == DUPLICATE)
         {
            startArray[i] = inputBegin(0);
            endArray[3*overlap-i-1] = inputEnd(0);
         }
      }

      //Copy values
      for(size_t i = overlap, j = 0; i < 3*overlap; ++i, ++j)
      {
         startArray[i] = inputBegin(j*stride);
      }
      InputIterator it = inputEnd-(((2*overlap)-1)*stride);
      for(size_t i = 0, j = 0; i < 2*overlap; ++i, ++j)
      {
         endArray[i] = it(j*stride);
      }
      size_t n = colWidth;
      m_mapOverlapFunc->setStride(1);
      for(size_t i = 0; i < overlap; ++i)
      {
         outputBegin(i*stride) = m_mapOverlapFunc->CPU(&startArray[i+overlap]);
      }
      m_mapOverlapFunc->setStride(stride);
      for(size_t i = overlap; i < n-overlap; ++i)
      {
         outputBegin(i*stride) = m_mapOverlapFunc->CPU(&(inputBegin(i*stride)));
      }
      m_mapOverlapFunc->setStride(1);
      for(size_t i= n-overlap; i<n; i++)
      {
         outputBegin(i*stride) = m_mapOverlapFunc->CPU(&endArray[overlap+i-(n-overlap)]);
      }
      inputBegin += 1; //colWidth;
      outputBegin += 1; //colWidth;
   }

   delete[] startArray;
   delete[] endArray;
}





}



