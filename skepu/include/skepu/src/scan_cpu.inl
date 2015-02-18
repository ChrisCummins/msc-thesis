/*! \file scan_cpu.inl
 *  \brief Contains the definitions of CPU specific member functions for the Scan skeleton.
 */

#include <iostream>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs the Scan on a whole Vector on the \em CPU with itself as output. A wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd, ScanType type, typename InputIterator::value_type init).
 *
 *  \param input A vector which will be scanned. It will be overwritten with the result.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::CPU(Vector<T>& input, ScanType type, T init)
{
   CPU(input.begin(), input.end(), type, init);
}


/*!
 *  Performs the row-wise Scan on a whole Matrix on the \em CPU with itself as output.
 *
 *  \param input A matrix which will be scanned. It will be overwritten with the result.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::CPU(Matrix<T>& input, ScanType type, T init)
{
   DEBUG_TEXT_LEVEL1("SCAN CPU Matrix\n")

   // Make sure we are properly synched with device data
   input.updateHostAndInvalidateDevice();

   T *data = input.getAddress();

   int rows = input.total_rows();
   int cols = input.total_cols();

   if(type == INCLUSIVE)
   {
      for(int r=0; r<rows; ++r)
      {
         for(int c=1; c<cols; ++c)
         {
            data[c] = m_scanFunc->CPU(data[c-1], data[c]);
         }
         data += cols;
      }
   }
   else
   {
      for(int r=0; r<rows; ++r)
      {
         T out_before = init;
         T out_current = data[0];

         data[0] = out_before;
         for(int c=1; c<cols; ++c)
         {
            out_before = data[c];
            data[c] = m_scanFunc->CPU(out_current, data[c-1]);
            T temp = out_before;
            out_before = out_current;
            out_current = temp;
         }
         data += cols;
      }
   }
}

/*!
 *  Performs the Scan on a range of elements on the \em CPU with the same range as output.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename InputIterator>
void Scan<ScanFunc>::CPU(InputIterator inputBegin, InputIterator inputEnd, ScanType type, typename InputIterator::value_type init)
{
   DEBUG_TEXT_LEVEL1("SCAN CPU\n")

   //Make sure we are properly synched with device data
   inputBegin.getParent().updateHostAndInvalidateDevice();

   // Uses operator() below instead of * or [] to avoid implicit synchronization and
   // unnecessary function calls.
   if(type == INCLUSIVE)
   {
      for(++inputBegin; inputBegin != inputEnd; ++inputBegin)
      {
         inputBegin(0) = m_scanFunc->CPU(inputBegin(-1), inputBegin(0));
      }
   }
   else
   {
      typename InputIterator::value_type out_before = init;
      typename InputIterator::value_type out_current = inputBegin(0);
      inputBegin(0) = out_before;
      for(++inputBegin; inputBegin != inputEnd; ++inputBegin)
      {
         out_before = inputBegin(0);
         inputBegin(0) = m_scanFunc->CPU(out_current, inputBegin(-1));
         typename InputIterator::value_type temp = out_before;
         out_before = out_current;
         out_current = temp;
      }
   }
}

/*!
 *  Performs the Scan on a whole Vector on the \em CPU with a seperate output vector. Wrapper for
 *  CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init).
 *
 *  \param input A vector which will be scanned.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::CPU(Vector<T>& input, Vector<T>& output, ScanType type, T init)
{
   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.size());
   }

   CPU(input.begin(), input.end(), output.begin(), type, init);
}




/*!
 *  Performs the row-wise Scan on a whole Matrix on the \em CPU with a separate Matrix as output.
 *
 *  \param input A matrix which will be scanned.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::CPU(Matrix<T>& input, Matrix<T>& output, ScanType type, T init)
{
   DEBUG_TEXT_LEVEL1("SCAN CPU Matrix\n")

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

   int rows = input.total_rows();
   int cols = input.total_cols();

   if(type == INCLUSIVE)
   {
      for(int r=0; r<rows; ++r)
      {
         out_data[0] = in_data[0];

         for(int c=1; c<cols; ++c)
         {
            out_data[c] = m_scanFunc->CPU(out_data[c-1], in_data[c]);
         }
         in_data += cols;
         out_data += cols;
      }
   }
   else
   {
      for(int r=0; r<rows; ++r)
      {
         out_data[0] = init;

         for(int c=1; c<cols; ++c)
         {
            out_data[c] = m_scanFunc->CPU(out_data[c-1], in_data[c-1]);
         }
         in_data += cols;
         out_data += cols;
      }
   }
}

/*!
 *  Performs the Scan on a range of elements on the \em CPU with a seperate output range.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename InputIterator, typename OutputIterator>
void Scan<ScanFunc>::CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init)
{
   DEBUG_TEXT_LEVEL1("SCAN CPU\n")

   //Make sure we are properly synched with device data
   outputBegin.getParent().invalidateDeviceData();
   inputBegin.getParent().updateHost();

   // Uses operator() below instead of * or [] to avoid implicit synchronization and
   // unnecessary function calls.
   if(type == INCLUSIVE)
   {
      outputBegin(0) = inputBegin(0);
      for(++inputBegin, ++outputBegin; inputBegin != inputEnd; ++inputBegin, ++outputBegin)
      {
         outputBegin(0) = m_scanFunc->CPU(outputBegin(-1), inputBegin(0));
      }
   }
   else
   {
      outputBegin(0) = init;
      for(++inputBegin, ++outputBegin; inputBegin != inputEnd; ++inputBegin, ++outputBegin)
      {
         outputBegin(0) = m_scanFunc->CPU(outputBegin(-1), inputBegin(-1));
      }
   }
}

}

