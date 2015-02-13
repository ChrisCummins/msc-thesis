/*! \file maparray_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the MapArray skeleton.
 */

#include <iostream>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs MapArray with the two vectors on the \em CPU. Seperate output is used.
 *  First Vector can be accessed entirely for each element in second Vector.
 *  The function is a wrapper for CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CPU(Vector<T>& input1, Vector<T>& input2, Vector<T>& output)
{
   if(input2.size() != output.size())
   {
      output.clear();
      output.resize(input2.size());
   }

   CPU(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin());
}



/*!
 *  Performs MapArray on the one vector and one matrix block-wise with \em CPU as backend. Seperate output vector is used.
 *  The Vector can be accessed entirely for "a block of elements" in the Matrix. The block-length is specified in the user-function.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CPU(Vector<T>& input1, Matrix<T>& input2, Vector<T>& output)
{
   size_t rows = input2.total_rows();
   size_t cols = input2.total_cols();

   size_t size = rows * cols;

   size_t p2BlockSize = m_mapArrayFunc->param2BlockSize;

   if( (size%p2BlockSize)!=0 )
   {
      SKEPU_ERROR("The 'p2BlockSize' specified in user function should be a perfect multiple of 'param2' size. Operation aborted.\n");
   }

   if((size/p2BlockSize) != output.size())
   {
      output.clear();
      output.resize(size/p2BlockSize);
   }

   //Make sure we are properly synched with device data
   input1.updateHost();
   input2.updateHost();

   output.invalidateDeviceData();

   size_t in2=0;

   size_t length = size/p2BlockSize;

   for(size_t i=0; i<length; i++)
   {
      output[i] = m_mapArrayFunc->CPU(&input1[0], &input2[in2]);
      in2 += p2BlockSize;
   }
}





/*!
 *  Performs MapArray on the one vector and one sparse matrix block-wise with \em CPU as backend. Seperate output vector is used.
 *  The Vector can be accessed entirely for "a block of elements" in the SparseMatrix. The block-length is specified in the user-function.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CPU(Vector<T>& input1, SparseMatrix<T>& input2, Vector<T>& output)
{
   size_t rows = input2.total_rows();
   size_t cols = input2.total_cols();

   size_t size = rows * cols;

   size_t p2BlockSize = m_mapArrayFunc->param2BlockSize;

   if( p2BlockSize!=cols )
   {
      SKEPU_ERROR("For Vector-SparseMatrix MapArray operation: The 'p2BlockSize' specified in user function should be equal to the number of columns in the sparse matrix.\n");
   }

   if((size/p2BlockSize) != output.size())
   {
      output.clear();
      output.resize(size/p2BlockSize);
   }

   //Make sure we are properly synched with device data
   input1.updateHost();
   input2.updateHost();

   output.invalidateDeviceData();

   size_t rowIdx = 0;
   size_t rowSize = 0;

   T *values = input2.get_values();
   size_t * row_offsets = input2.get_row_pointers();
   size_t * col_indices = input2.get_col_indices();

   size_t nzz = input2.total_nnz();

   for(size_t r=0; r<rows; ++r)
   {
      rowIdx = row_offsets[r];
      rowSize = row_offsets[r+1]-rowIdx;

      output[r] = m_mapArrayFunc->CPU(&input1[0], &values[rowIdx], rowSize, &col_indices[rowIdx]);
   }
}







/*!
 *  Performs MapArray with the one vector and one matrix on the \em CPU. Seperate output is used.
 *  The Vector can be accessed entirely for each element in Matrix.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CPU(Vector<T>& input1, Matrix<T>& input2, Matrix<T>& output)
{
   DEBUG_TEXT_LEVEL1("MAPARRAY CPU\n")

   if(input2.size() != output.size())
   {
      output.clear();
      output.resize(input2.total_rows(), input2.total_cols());
   }

   size_t rows = input2.total_rows();
   size_t cols = input2.total_cols();

   //Make sure we are properly synched with device data
   input1.updateHost();
   input2.updateHost();

   output.invalidateDeviceData();

   for(size_t i=0; i<rows; i++)
   {
      for(size_t j=0; j<cols; j++)
      {
         output[i*cols+j] = m_mapArrayFunc->CPU(&input1[0], input2[i*cols+j], j, i);
      }
   }
}

/*!
 *  Performs the MapArray with the two element ranges on the \em CPU. Seperate output range is used.
 *  First range can be accessed entirely for each element in second range.
 *  The computation is done by iterating over all the elements in the ranges
 *  and applying the user function on each of them.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param outputBegin An iterator to the first element of the output range.
 */
template <typename MapArrayFunc>
template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
void MapArray<MapArrayFunc>::CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin)
{
   DEBUG_TEXT_LEVEL1("MAPARRAY CPU\n")

   //Make sure we are properly synched with device data
   input1Begin.getParent().updateHost();
   input2Begin.getParent().updateHost();

   outputBegin.getParent().invalidateDeviceData();

   for(; input2Begin != input2End; ++input2Begin, ++outputBegin)
   {
      outputBegin(0) = m_mapArrayFunc->CPU(&input1Begin(0), input2Begin(0));
   }
}

}

