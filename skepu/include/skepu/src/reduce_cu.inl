/*! \file reduce_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the Reduce skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>

#include "operator_type.h"

#include "reduce_kernels.h"
#include "device_mem_pointer_cu.h"
#include "device_cu.h"

namespace skepu
{



/*!
 *  Performs the Reduction on a range of elements with \em CUDA as backend. Returns a scalar result. The function
 *  uses only \em one device which is decided by a parameter. A Helper method.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type Reduce<ReduceFunc, ReduceFunc>::reduceSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID)
{
   cudaSetDevice(deviceID);

   // Setup parameters
   size_t size = inputEnd-inputBegin;

   size_t maxThreads = 512;  // number of threads per block, taken from NVIDIA source
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

   typename InputIterator::value_type result = 0;

   // Copies "all" elements to the device at once, better?
   typename InputIterator::device_pointer_type_cu in_mem_p = inputBegin.getParent().updateDevice_CU(inputBegin.getAddress(), size, deviceID, true, false);
   DeviceMemPointer_CU<typename InputIterator::value_type> out_mem_p(&result, numBlocks, m_environment->m_devices_CU.at(deviceID));


#ifdef USE_PINNED_MEMORY
   ExecuteReduceOnADevice<ReduceFunc, typename InputIterator::value_type>(m_reduceFunc, size, numThreads, numBlocks, maxThreads, maxBlocks, in_mem_p->getDeviceDataPointer(), out_mem_p.getDeviceDataPointer(), deviceID, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]));
#else
   ExecuteReduceOnADevice<ReduceFunc, typename InputIterator::value_type>(m_reduceFunc, size, numThreads, numBlocks, maxThreads, maxBlocks, in_mem_p->getDeviceDataPointer(), out_mem_p.getDeviceDataPointer(), deviceID);
#endif

   out_mem_p.changeDeviceData();
   out_mem_p.copyDeviceToHost(1);

#ifdef USE_PINNED_MEMORY // ensure synchronization...
   cutilDeviceSynchronize(); //Do CUTIL way, more safe approach, if result is incorrect could move it up.....
#endif

   return result;
}







/*!
 *  Performs the Reduction on a SparseMatrix with \em CUDA as backend. Returns a scalar result. The function
 *  uses only \em one device which is decided by a parameter. A Helper method.
 *
 *  \param input A sparse input sparse matrix on which reduction is performed.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::reduceSingleThread_CU(SparseMatrix<T> &input, unsigned int deviceID)
{
   cudaSetDevice(deviceID);

   // Setup parameters
   size_t size = input.total_nnz();

   size_t maxThreads = 512;  // number of threads per block, taken from NVIDIA source
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   getNumBlocksAndThreads(size, maxBlocks, maxThreads, numBlocks, numThreads);

   T result = 0;

   // Copies "all" elements to the device at once, better?
   typename SparseMatrix<T>::device_pointer_type_cu in_mem_p = input.updateDevice_CU(input.get_values(), size, deviceID, true);
   DeviceMemPointer_CU<T> out_mem_p(&result, numBlocks, m_environment->m_devices_CU.at(deviceID));

#ifdef USE_PINNED_MEMORY
   ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, size, numThreads, numBlocks, maxThreads, maxBlocks, in_mem_p->getDeviceDataPointer(), out_mem_p.getDeviceDataPointer(), deviceID, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]));
#else
   ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, size, numThreads, numBlocks, maxThreads, maxBlocks, in_mem_p->getDeviceDataPointer(), out_mem_p.getDeviceDataPointer(), deviceID);
#endif

   out_mem_p.changeDeviceData();
   out_mem_p.copyDeviceToHost(1);

#ifdef USE_PINNED_MEMORY // ensure synchronization...
   cutilDeviceSynchronize(); //Do CUTIL way, more safe approach, if result is incorrect could move it up.....
#endif

   return result;
}






/*!
 *  Performs the Reduction on a whole Vector. Returns a scalar result. A wrapper for CU(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU).
 *  Using \em CUDA as backend.
 *
 *  \param input A vector which the reduction will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::CU(Vector<T>& input, int useNumGPU)
{
   return CU(input.begin(), input.end(), useNumGPU);
}



/*!
 *  Performs the Reduction on a whole Matrix. Returns a scalar result. A wrapper for CU(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU).
 *  Using \em CUDA as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::CU(Matrix<T>& input, int useNumGPU)
{
   return CU(input.begin(), input.end(), useNumGPU);
}


/*!
 *  Performs the Reduction on non-zero elements of a SparseMatrix. Returns a scalar result.
 *  The function decides whether to perform the reduction on one device, calling
 *  reduceSingleThread_CU(SparseMatrix<T> &input, int deviceID) or
 *  on multiple devices, dividing the range of elements equally among the participating devices each reducing
 *  its part. The results are then reduced themselves on the CPU.
 *  Using \em CUDA as backend.
 *
 *  \param input A sparse matrix which the reduction will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::CU(SparseMatrix<T> &input, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE SparseMatrix CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      return reduceSingleThread_CU(input, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t n = input.total_nnz();

      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      T result[MAX_GPU_DEVICES];

      typename SparseMatrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES]; // as vector, matrix and sparse matrix have common types
      typename SparseMatrix<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // Setup parameters
      size_t maxThreads = 512;
      size_t maxBlocks = 64;

      size_t numThreads[MAX_GPU_DEVICES];
      size_t numBlocks[MAX_GPU_DEVICES];

      T *values = input.get_values();

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         numBlocks[i] = numThreads[i] = 0;

         getNumBlocksAndThreads(numElem, maxBlocks, maxThreads, numBlocks[i], numThreads[i]);

         in_mem_p[i] = input.updateDevice_CU((values+i*numElemPerSlice), numElem, i, false);

         // Create the output memory
         out_mem_p[i] = new DeviceMemPointer_CU<T>(&result[i], numBlocks[i], m_environment->m_devices_CU.at(i));
      }

      // Create argument structs for all threads
      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in_mem_p[i] = input.updateDevice_CU((values+i*numElemPerSlice), numElem, i, true);

#ifdef USE_PINNED_MEMORY
         ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, numElem, numThreads[i], numBlocks[i], maxThreads, maxBlocks, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), i, (m_environment->m_devices_CU.at(i)->m_streams[0]));
#else
         ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, numElem, numThreads[i], numBlocks[i], maxThreads, maxBlocks, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), i);
#endif

         //Just mark data change
         out_mem_p[i]->changeDeviceData();

#ifdef USE_PINNED_MEMORY // then make copy as it is asynchronous		    	
         out_mem_p[i]->copyDeviceToHost(1);
#endif
      }

      // Joins the threads and reduces the results on the CPU, yielding the total result.
      cudaSetDevice(0);

#ifdef USE_PINNED_MEMORY // if pinned, just synchornize
      cutilDeviceSynchronize();
#else 					  // in normal case copy here...
      out_mem_p[0]->copyDeviceToHost(1);
#endif

      T totalResult = result[0];
      delete out_mem_p[0];

      for(size_t i = 1; i < numDevices; ++i)
      {
         cudaSetDevice(i);

#ifdef USE_PINNED_MEMORY // if pinned, just synchornize
         cutilDeviceSynchronize();
#else 					  // in normal case copy here...
         out_mem_p[i]->copyDeviceToHost(1);
#endif

         totalResult = m_reduceFunc->CPU(totalResult, result[i]);

         delete out_mem_p[i];
      }

      cudaSetDevice(m_environment->bestCUDADevID);
      return totalResult;
   }
}




/*!
 *  Performs the Reduction on a range of elements. Returns a scalar result. The function decides whether to perform
 *  the reduction on one device, calling reduceSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, int deviceID) or
 *  on multiple devices, dividing the range of elements equally among the participating devices each reducing
 *  its part. The results are then reduced themselves on the CPU.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type Reduce<ReduceFunc, ReduceFunc>::CU(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      return reduceSingleThread_CU(inputBegin, inputEnd, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t n = inputEnd - inputBegin;

      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename InputIterator::value_type result[MAX_GPU_DEVICES];

      typename InputIterator::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];

      typename InputIterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // Setup parameters
      size_t maxThreads = 512;
      size_t maxBlocks = 64;

      size_t numThreads[MAX_GPU_DEVICES];
      size_t numBlocks[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         numBlocks[i] = numThreads[i] = 0;

         getNumBlocksAndThreads(numElem, maxBlocks, maxThreads, numBlocks[i], numThreads[i]);

         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numElem, i, false, false);

         // Create the output memory
         out_mem_p[i] = new DeviceMemPointer_CU<typename InputIterator::value_type>(&result[i], numBlocks[i], m_environment->m_devices_CU.at(i));
      }


      // Create argument structs for all threads
      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numElem, i, true, false);

#ifdef USE_PINNED_MEMORY
         ExecuteReduceOnADevice<ReduceFunc, typename InputIterator::value_type>(m_reduceFunc, numElem, numThreads[i], numBlocks[i], maxThreads, maxBlocks, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), i, (m_environment->m_devices_CU.at(i)->m_streams[0]));
#else
         ExecuteReduceOnADevice<ReduceFunc, typename InputIterator::value_type>(m_reduceFunc, numElem, numThreads[i], numBlocks[i], maxThreads, maxBlocks, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), i);
#endif

         //Just mark data change
         out_mem_p[i]->changeDeviceData();

#ifdef USE_PINNED_MEMORY // then make copy as it is asynchronous		    	
         out_mem_p[i]->copyDeviceToHost(1);
#endif
      }

      // Joins the threads and reduces the results on the CPU, yielding the total result.
      cudaSetDevice(0);

#ifdef USE_PINNED_MEMORY // if pinned, just synchornize
      cutilDeviceSynchronize();
#else 					  // in normal case copy here...
      out_mem_p[0]->copyDeviceToHost(1);
#endif

      typename InputIterator::value_type totalResult = result[0];
      delete out_mem_p[0];

      for(size_t i = 1; i < numDevices; ++i)
      {
         cudaSetDevice(i);

#ifdef USE_PINNED_MEMORY // if pinned, just synchornize
         cutilDeviceSynchronize();
#else 					  // in normal case copy here...
         out_mem_p[i]->copyDeviceToHost(1);
#endif

         totalResult = m_reduceFunc->CPU(totalResult, result[i]);

         delete out_mem_p[i];
      }

      cudaSetDevice(m_environment->bestCUDADevID);
      return totalResult;
   }
}






















/*!
 *  Performs the Reduction on a Matrix with \em CUDA as backend either row-wise or column-wise. Returns a scalar result. The function
 *  uses only \em one device which is decided by a parameter. A Helper method.
 *
 *  \param input An input matrix on which reduction is performed.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
void Reduce<ReduceFunc, ReduceFunc>::reduceSingleThreadOneDim_CU(Matrix<T> &input, unsigned int deviceID, skepu::Vector<T> &result)
{
   cudaSetDevice(deviceID);

   Device_CU *device = m_environment->m_devices_CU.at(deviceID);
   unsigned int maxKernelsSupported = device->getNoConcurrentKernels();

   // Setup parameters
   size_t rows = input.total_rows();
   size_t cols = input.total_cols();
   size_t size = rows*cols;

   size_t maxThreads = 512;  // number of threads per block, taken from NVIDIA source
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   getNumBlocksAndThreads(cols, maxBlocks, maxThreads, numBlocks, numThreads); // first do it for each column

   // Copies "all" elements to the device at once, better?
   typename Matrix<T>::device_pointer_type_cu in_mem_p = input.updateDevice_CU(input.getAddress(), size, deviceID, true, false);

   // Manually allocate output memory in this case,
   T *deviceMemPointer;
   allocateCudaMemory<T>(&deviceMemPointer, rows*numBlocks);

   T *d_input = in_mem_p->getDeviceDataPointer();
   T *d_output = deviceMemPointer;

   // First reduce all elements row-wise so that each row produces one element.
   for(size_t r=0; r<rows; r++)
   {
#ifdef USE_PINNED_MEMORY
      ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID, (device->m_streams[(r%maxKernelsSupported)]));
#else
      ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID);
#endif

#ifdef USE_PINNED_MEMORY
      copyDeviceToHost(&result[r], d_output, 1, (device->m_streams[(r%maxKernelsSupported)]) );
#else
      copyDeviceToHost(&result[r], d_output, 1);
#endif

      d_input += cols;
      d_output += numBlocks;
   }

   cutilDeviceSynchronize(); //Do CUTIL way, more safe approach,

   freeCudaMemory<T>(deviceMemPointer);
}


/*!
 *  Performs the Reduction on a whole Matrix either row-wise or column-wise. Returns a \em SkePU vector of reduction result.
 *  Using \em CUDA as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \param reducePolicy The policy specifying how reduction will be performed, can be either REDUCE_ROW_WISE_ONLY of REDUCE_COL_WISE_ONLY
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
skepu::Vector<T> Reduce<ReduceFunc, ReduceFunc>::CU(Matrix<T> &input, ReducePolicy reducePolicy, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE Matrix[] CUDA\n")

   Matrix<T> *matrix = NULL;

   if(reducePolicy==REDUCE_COL_WISE_ONLY)
      matrix = &(~input);
   else // assume  reducePolict==REDUCE_ROW_WISE_ONLY)
      matrix = &input;

   size_t rows = matrix->total_rows();
   size_t cols = matrix->total_cols();

   skepu::Vector<T> result(rows);

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      reduceSingleThreadOneDim_CU(*matrix, m_environment->bestCUDADevID, result);
      return result;
   }
   else
   {
      size_t maxThreads = 512;  // number of threads per block, taken from NVIDIA source
      size_t maxBlocks = 64;

      size_t numRowsPerSlice = rows / numDevices;
      size_t restRows = rows % numDevices;

      typename Matrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];

      T *deviceMemPointers[MAX_GPU_DEVICES];

      // Setup parameters
      size_t numThreads = 0;
      size_t numBlocks = 0;

      getNumBlocksAndThreads(cols, maxBlocks, maxThreads, numBlocks, numThreads); // first do it for each column

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numRows;
         if(i == numDevices-1)
            numRows = numRowsPerSlice+restRows;
         else
            numRows = numRowsPerSlice;

         in_mem_p[i] = matrix->updateDevice_CU((matrix->getAddress()+i*numRowsPerSlice*cols), numRows*cols, i, false, false);


         size_t outSize=numRows*numBlocks;

         // Manually allocate output memory in this case
         allocateCudaMemory<T>(&deviceMemPointers[i], outSize);
      }

      T *d_input = NULL;
      T *d_output = NULL;

      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         Device_CU *device = m_environment->m_devices_CU.at(i);
         unsigned int maxKernelsSupported = device->getNoConcurrentKernels();

         size_t numRows;
         if(i == numDevices-1)
            numRows = numRowsPerSlice+restRows;
         else
            numRows = numRowsPerSlice;

         in_mem_p[i] = matrix->updateDevice_CU((matrix->getAddress()+i*numRowsPerSlice*cols), numRows*cols, i, true, false);

         d_input = in_mem_p[i]->getDeviceDataPointer();
         d_output = deviceMemPointers[i];

         // Reduce all elements row-wise so that each row produces one element.
         for(size_t r=0; r<numRows; r++)
         {
#ifdef USE_PINNED_MEMORY
            ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i, (device->m_streams[(r%maxKernelsSupported)]));
#else
            ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i);
#endif

#ifdef USE_PINNED_MEMORY
            copyDeviceToHost(&result[r+(numRowsPerSlice*i)], d_output, 1, (device->m_streams[(r%maxKernelsSupported)]) );
#else
            copyDeviceToHost(&result[r+(numRowsPerSlice*i)], d_output, 1);
#endif

            if(r!=numRows-1)
            {
               d_input += cols;
               d_output += numBlocks;
            }
         }
      }

      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);
         cutilDeviceSynchronize();

         freeCudaMemory<T>(deviceMemPointers[i]);
      }

      return result;
   }
}











/*!
 *  Performs the Reduction, either row-wise or column-wise, on non-zero elements of a SparseMatrix. Has an output parameter \em SkePU vector of reduction result.
 *  The function uses \em CUDA as backend with only \em one device which is decided by a parameter. A Helper method.
 *
 *  \param input A sparse input matrix on which reduction is performed.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
void Reduce<ReduceFunc, ReduceFunc>::reduceSingleThreadOneDim_CU(SparseMatrix<T> &input, unsigned int deviceID, skepu::Vector<T> &result)
{
   cudaSetDevice(deviceID);

   Device_CU *device = m_environment->m_devices_CU.at(deviceID);
   unsigned int maxKernelsSupported = device->getNoConcurrentKernels();

   T *resultPtr = result.getAddress();

   // Setup parameters
   size_t rows = input.total_rows();
   size_t size = input.total_nnz();

   size_t avgElemPerRow = size/rows;

   size_t maxThreads = 512;  // number of threads per block, taken from NVIDIA source
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   getNumBlocksAndThreads(avgElemPerRow, maxBlocks, maxThreads, numBlocks, numThreads); // do it for each column

   // Copies "all" elements to the device at once, better?
   typename SparseMatrix<T>::device_pointer_type_cu in_mem_p = input.updateDevice_CU(input.get_values(), size, deviceID, true);

   // Manually allocate output memory in this case, if only 1 block allocate for two
   T *deviceMemPointer;
   allocateCudaMemory<T>(&deviceMemPointer, (rows*numBlocks));

   T *d_input = in_mem_p->getDeviceDataPointer();
   T *d_output = deviceMemPointer;

   // First reduce all elements row-wise so that each row produces one element.
   for(size_t r=0; r<rows; r++)
   {
      size_t elemPerRow = input.get_rowSize(r);

      if(elemPerRow>1)
      {
#ifdef USE_PINNED_MEMORY
         ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, elemPerRow, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID, (device->m_streams[(r%maxKernelsSupported)]), false);
#else
         ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, elemPerRow, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID, false);
#endif

#ifdef USE_PINNED_MEMORY
         copyDeviceToHost(&resultPtr[r], d_output, 1, (device->m_streams[(r%maxKernelsSupported)]) );
#else
         copyDeviceToHost(&resultPtr[r], d_output, 1);
#endif
      }
      else
         resultPtr[r] = ((elemPerRow>0) ? (input.begin(r)(0)):T()); // dont use [] operator as that internally invalidate device copy

      d_input += elemPerRow;
      d_output += numBlocks;
   }

   cutilDeviceSynchronize(); //Do CUTIL way, more safe approach,

   freeCudaMemory<T>(deviceMemPointer);
}




/*!
 *  Performs the Reduction, either row-wise or column-wise, on non-zero elements of a SparseMatrix. Returns a \em SkePU vector of reduction result.
 *  Using \em CUDA as backend.
 *
 *  \param input A sparse matrix which the reduction will be performed on.
 *  \param reducePolicy The policy specifying how reduction will be performed, can be either REDUCE_ROW_WISE_ONLY of REDUCE_COL_WISE_ONLY
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
skepu::Vector<T> Reduce<ReduceFunc, ReduceFunc>::CU(SparseMatrix<T> &input, ReducePolicy reducePolicy, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE SparseMatrix[] CUDA\n")

   SparseMatrix<T> *matrix = NULL;

   if(reducePolicy==REDUCE_COL_WISE_ONLY)
      matrix = &(~input);
   else // assume  reducePolict==REDUCE_ROW_WISE_ONLY)
      matrix = &input;

   size_t rows = matrix->total_rows();

   skepu::Vector<T> result(rows);

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      reduceSingleThreadOneDim_CU(*matrix, m_environment->bestCUDADevID, result);
      return result;
   }
   else
   {
      T *resultPtr= result.getAddress();

      size_t size = matrix->total_nnz();

      size_t avgElemPerRow = size/rows;

      size_t maxThreads = 512;  // number of threads per block, taken from NVIDIA source
      size_t maxBlocks = 64;

      size_t numRowsPerSlice = rows / numDevices;
      size_t restRows = rows % numDevices;

      typename SparseMatrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];

      T *deviceMemPointers[MAX_GPU_DEVICES];

      // Setup parameters
      size_t numThreads = 0;
      size_t numBlocks = 0;

      getNumBlocksAndThreads(avgElemPerRow, maxBlocks, maxThreads, numBlocks, numThreads); // first do it for each column

      size_t offset = 0;
      size_t end = 0;

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numRows;
         if(i == numDevices-1)
            numRows = numRowsPerSlice+restRows;
         else
            numRows = numRowsPerSlice;


         end = matrix->get_rowOffsetFromStart(numRows+i*numRowsPerSlice);

         in_mem_p[i] = matrix->updateDevice_CU((matrix->get_values()+offset), end-offset, i, false);

         size_t outSize=numRows*numBlocks;

         // Manually allocate output memory in this case
         allocateCudaMemory<T>(&deviceMemPointers[i], outSize);

         offset = end;
      }

      T *d_input = NULL;
      T *d_output = NULL;

      offset = end = 0;

      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         Device_CU *device = m_environment->m_devices_CU.at(i);
         unsigned int maxKernelsSupported = device->getNoConcurrentKernels();

         size_t numRows;
         if(i == numDevices-1)
            numRows = numRowsPerSlice+restRows;
         else
            numRows = numRowsPerSlice;

         end = matrix->get_rowOffsetFromStart(numRows+i*numRowsPerSlice);

         in_mem_p[i] = matrix->updateDevice_CU((matrix->get_values()+offset), end-offset, i, true);

         d_input = in_mem_p[i]->getDeviceDataPointer();
         d_output = deviceMemPointers[i];

         // First reduce all elements row-wise so that each row produces one element.
         for(size_t r=0; r<numRows; r++)
         {
            size_t elemPerRow = matrix->get_rowSize(r+(numRowsPerSlice*i));

            if(elemPerRow>1)
            {
#ifdef USE_PINNED_MEMORY
               ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, elemPerRow, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i, (device->m_streams[(r%maxKernelsSupported)]), false);
#else
               ExecuteReduceOnADevice<ReduceFunc, T>(m_reduceFunc, elemPerRow, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i, false);
#endif

#ifdef USE_PINNED_MEMORY
               copyDeviceToHost(&resultPtr[r+(numRowsPerSlice*i)], d_output, 1, (device->m_streams[(r%maxKernelsSupported)]) );
#else
               copyDeviceToHost(&resultPtr[r+(numRowsPerSlice*i)], d_output, 1);
#endif
            }
            else
               resultPtr[r+(numRowsPerSlice*i)] = ((elemPerRow>0) ? (matrix->begin(r+(numRowsPerSlice*i))(0)):T()); // dont use [] operator as that internally invalidate device copy

            d_input += elemPerRow;
            d_output += numBlocks;
         }

         offset = end;
      }

      // Free allocated memory on all devices.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);
         cutilDeviceSynchronize();

         freeCudaMemory<T>(deviceMemPointers[i]);
      }

      return result;
   }
}





}

#endif

