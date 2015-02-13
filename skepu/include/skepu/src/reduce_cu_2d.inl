/*! \file reduce_cu_2d.inl
 *  \brief Contains the definitions of CUDA specific member functions for the 2DReduce skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
//#include <cutil.h>
#include <iostream>

#include "operator_type.h"

#include "reduce_kernels.h"
#include "device_mem_pointer_cu.h"
#include "device_cu.h"

namespace skepu
{


/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  matrix by using \em CUDA backend. Returns a scalar result. The function
 *  uses only \em one CUDA device which is decided by a parameter.
 *
 *  \param input An input matrix whose elements need to be reduced.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::reduceSingleThread_CU(Matrix<T>& input, unsigned int deviceID)
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

   std::vector<T, malloc_allocator<T> > tempResult(rows);

   // Copies "all" elements to the device at once, better?
   typename Matrix<T>::device_pointer_type_cu in_mem_p = input.updateDevice_CU(input.getAddress(), size, deviceID, true, false);

   cutilSafeCall(cudaStreamSynchronize(device->m_streams[0]));

   // Manually allocate output memory in this case, if only 1 block allocate for two
   T *deviceMemPointer;
   allocateCudaMemory<T>(&deviceMemPointer, rows*((numBlocks>1)?numBlocks:2));

   T *d_input = in_mem_p->getDeviceDataPointer();
   T *d_output = deviceMemPointer;

   // First reduce all elements row-wise so that each row produces one element.
   for(size_t r=0; r<rows; r++)
   {
#ifdef USE_PINNED_MEMORY
      ExecuteReduceOnADevice<ReduceFuncRowWise, T>(m_reduceFuncRowWise, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID, (device->m_streams[(r%maxKernelsSupported)]));
#else
      ExecuteReduceOnADevice<ReduceFuncRowWise, T>(m_reduceFuncRowWise, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID);
#endif

#ifdef USE_PINNED_MEMORY
      copyDeviceToHost(&tempResult[r], d_output, 1, (device->m_streams[(r%maxKernelsSupported)]) );
#else
      copyDeviceToHost(&tempResult[r], d_output, 1);
#endif

      d_input += cols;
      d_output += numBlocks;
   }

   T result;

   cutilDeviceSynchronize(); // Synchronize the device to ensure that all intermediate results are available

   // if sufficient work then do final (column-wise) reduction on GPU
   if(rows>REDUCE_GPU_THRESHOLD)
   {
      // reset to starting position and use it as an input
      d_input = deviceMemPointer;

      d_output = deviceMemPointer+rows; // re-use already allocated space as well for output.

#ifdef USE_PINNED_MEMORY
      copyHostToDevice(&tempResult[0], d_input, rows, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]) );
#else
      copyHostToDevice(&tempResult[0], d_input, rows);
#endif

      getNumBlocksAndThreads(rows, maxBlocks, maxThreads, numBlocks, numThreads); // get numThreads and numBlocks for final reduction

#ifdef USE_PINNED_MEMORY
      ExecuteReduceOnADevice<ReduceFuncColWise, T>(m_reduceFuncColWise, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]));
#else
      ExecuteReduceOnADevice<ReduceFuncColWise, T>(m_reduceFuncColWise, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID);
#endif

#ifdef USE_PINNED_MEMORY
      copyDeviceToHost(&result, d_output, 1, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]) );
#else
      copyDeviceToHost(&result, d_output, 1);
#endif

      cutilDeviceSynchronize();
   }
   else // do final reduction step on CPU instead
   {
      result = tempResult[0];

      for(size_t r=1; r<rows; r++)
      {
         result = m_reduceFuncColWise->CPU(result, tempResult[r]);
      }
   }

   freeCudaMemory<T>(deviceMemPointer);

   return result;
}




/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  matrix by using \em CUDA backend. Returns a scalar result. The function
 *  uses a variable number of devices, dividing the range of elemets equally
 *  among the participating devices each reducing its part. The results are
 *  then reduced themselves on the CPU.
 *
 *  \param input An input matrix whose elements need to be reduced.
 *  \param numDevices Integer deciding how many devices to utilize.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::reduceMultipleThreads_CU(Matrix<T>& input, size_t numDevices)
{
   size_t rows = input.total_rows();
   size_t cols = input.total_cols();

   size_t maxThreads = 512;  // number of threads per block, taken from NVIDIA source
   size_t maxBlocks = 64;

   size_t numRowsPerSlice = rows / numDevices;
   size_t restRows = rows % numDevices;

   typename Matrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];

   T *deviceMemPointers[MAX_GPU_DEVICES];

   // Setup parameters
   size_t numThreads = 0;
   size_t numBlocks = 0; //[MAX_GPU_DEVICES];

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

      in_mem_p[i] = input.updateDevice_CU((input.getAddress()+i*numRowsPerSlice*cols), numRows*cols, i, false, false);


      size_t outSize=numRows*numBlocks;
      if(i==0 && outSize<(2*rows)) // for first device as later we may re-use this storage to do final reduction on GPU 0
         outSize = 2*rows; // make it at least that much large

      // Manually allocate output memory in this case
      allocateCudaMemory<T>(&deviceMemPointers[i], outSize);
   }

   std::vector<T, malloc_allocator<T> > tempResult(rows);

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

      in_mem_p[i] = input.updateDevice_CU((input.getAddress()+i*numRowsPerSlice*cols), numRows*cols, i, true, false);

      d_input = in_mem_p[i]->getDeviceDataPointer();
      d_output = deviceMemPointers[i];

      // First reduce all elements row-wise so that each row produces one element.
      for(size_t r=0; r<numRows; r++)
      {
#ifdef USE_PINNED_MEMORY
         ExecuteReduceOnADevice<ReduceFuncRowWise, T>(m_reduceFuncRowWise, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i, (device->m_streams[(r%maxKernelsSupported)]));
#else
         ExecuteReduceOnADevice<ReduceFuncRowWise, T>(m_reduceFuncRowWise, cols, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i);
#endif

#ifdef USE_PINNED_MEMORY
         copyDeviceToHost(&tempResult[r+(numRowsPerSlice*i)], d_output, 1, (m_environment->m_devices_CU.at(i)->m_streams[(r%maxKernelsSupported)]) );
#else
         copyDeviceToHost(&tempResult[r+(numRowsPerSlice*i)], d_output, 1);
#endif

         if(r!=numRows-1)
         {
            d_input += cols;
            d_output += numBlocks;
         }
      }
   }

   // Synchronize all devices (?)
   m_environment->finishAll_CU(0, numDevices);

   T result;

   // if sufficient work then do final (column-wise) reduction on GPU
   if(rows>REDUCE_GPU_THRESHOLD)
   {
      cudaSetDevice(m_environment->bestCUDADevID); // do it on a single GPU or a CPU, should not be that much work(?)

      // reset to starting position and use it as an input
      d_input = deviceMemPointers[0];

      d_output = deviceMemPointers[0]+rows; // re-use already allocated space as well for output.

#ifdef USE_PINNED_MEMORY
      copyHostToDevice(&tempResult[0], d_input, rows, (m_environment->m_devices_CU.at(m_environment->bestCUDADevID)->m_streams[0]) );
#else
      copyHostToDevice(&tempResult[0], d_input, rows);
#endif

      getNumBlocksAndThreads(rows, maxBlocks, maxThreads, numBlocks, numThreads); // get numThreads and numBlocks for final reduction

#ifdef USE_PINNED_MEMORY
      ExecuteReduceOnADevice<ReduceFuncColWise, T>(m_reduceFuncColWise, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, m_environment->bestCUDADevID, (m_environment->m_devices_CU.at(m_environment->bestCUDADevID)->m_streams[0]));
#else
      ExecuteReduceOnADevice<ReduceFuncColWise, T>(m_reduceFuncColWise, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, m_environment->bestCUDADevID);
#endif

#ifdef USE_PINNED_MEMORY
      copyDeviceToHost(&result, d_output, 1, (m_environment->m_devices_CU.at(m_environment->bestCUDADevID)->m_streams[0]) );
#else
      copyDeviceToHost(&result, d_output, 1);
#endif

      cutilDeviceSynchronize();
   }
   else // do final reduction step on CPU instead
   {
      result = tempResult[0];

      for(size_t r=1; r<rows; r++)
      {
         result = m_reduceFuncColWise->CPU(result, tempResult[r]);
      }
   }

   // Free allocated memory on all devices, some pathetic issue, gives error when try to clear all poiters except first one
   for(size_t i = 0; i < numDevices; ++i)
   {
      cudaSetDevice(i);
      freeCudaMemory<T>(deviceMemPointers[i]);
   }

   return result;
}



/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  matrix by using \em CUDA backend. Returns a scalar result. The function
 *  can be applied by any number of CUDA devices, thus internally calling the
 *  \em reduceSingle_CL or \em reduceNumDevices_CL depending upon number of
 *  CUDA devices specified/available.
 *
 *  \param input An input matrix whose elements need to be reduced.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::CU(Matrix<T>& input, int useNumGPU)
{
   cudaGetLastError();

   DEBUG_TEXT_LEVEL1("REDUCE 2D Matrix CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;

   if(numDevices < 2)
      return reduceSingleThread_CU(input, (m_environment->bestCUDADevID));
   else
      return reduceMultipleThreads_CU(input, numDevices);
}
















/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  matrix by using \em CUDA backend. Returns a scalar result. The function
 *  uses only \em one CUDA device which is decided by a parameter.
 *
 *  \param input An input matrix whose elements need to be reduced.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::reduceSingleThread_CU(SparseMatrix<T>& input, unsigned int deviceID)
{
   cudaSetDevice(deviceID);

   Device_CU *device = m_environment->m_devices_CU.at(deviceID);
   unsigned int maxKernelsSupported = device->getNoConcurrentKernels();

   // Setup parameters
   size_t rows = input.total_rows();
   size_t size = input.total_nnz();

   size_t avgElemPerRow = size/rows;

   size_t maxThreads = 512;  // number of threads per block, taken from NVIDIA source
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   getNumBlocksAndThreads(avgElemPerRow, maxBlocks, maxThreads, numBlocks, numThreads); // first do it for each column

   std::vector<T, malloc_allocator<T> > tempResult(rows);

   // Copies "all" elements to the device at once, better?
   typename SparseMatrix<T>::device_pointer_type_cu in_mem_p = input.updateDevice_CU(input.get_values(), size, deviceID, true);

   // Manually allocate output memory in this case, if only 1 block allocate for two
   T *deviceMemPointer;
   allocateCudaMemory<T>(&deviceMemPointer, rows*((numBlocks>1)?numBlocks:2));

   T *d_input = in_mem_p->getDeviceDataPointer();
   T *d_output = deviceMemPointer;

   // First reduce all elements row-wise so that each row produces one element.
   for(size_t r=0; r<rows; r++)
   {
      size_t elemPerRow = input.get_rowSize(r);

      if(elemPerRow>1)
      {
#ifdef USE_PINNED_MEMORY
         ExecuteReduceOnADevice<ReduceFuncRowWise, T>(m_reduceFuncRowWise, elemPerRow, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID, (device->m_streams[(r%maxKernelsSupported)]));
#else
         ExecuteReduceOnADevice<ReduceFuncRowWise, T>(m_reduceFuncRowWise, elemPerRow, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID);
#endif

#ifdef USE_PINNED_MEMORY
         copyDeviceToHost(&tempResult[r], d_output, 1, (device->m_streams[(r%maxKernelsSupported)]) );
#else
         copyDeviceToHost(&tempResult[r], d_output, 1);
#endif
      }
      else
         tempResult[r] = ((elemPerRow>0) ? (input.begin(r)(0)):T()); // dont use [] operator as that internally invalidate device copy

      d_input += elemPerRow;
      d_output += numBlocks;
   }

   T result;

   cutilDeviceSynchronize(); // Synchronize the device to ensure that all intermediate results are available

   // if sufficient work then do final (column-wise) reduction on GPU
   if(rows>REDUCE_GPU_THRESHOLD)
   {
      // reset to starting position and use it as an input
      d_input = deviceMemPointer;

      d_output = deviceMemPointer+rows; // re-use already allocated space as well for output.

#ifdef USE_PINNED_MEMORY
      copyHostToDevice(&tempResult[0], d_input, rows, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]) );
#else
      copyHostToDevice(&tempResult[0], d_input, rows);
#endif

      getNumBlocksAndThreads(rows, maxBlocks, maxThreads, numBlocks, numThreads); // get numThreads and numBlocks for final reduction

#ifdef USE_PINNED_MEMORY
      ExecuteReduceOnADevice<ReduceFuncColWise, T>(m_reduceFuncColWise, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]));
#else
      ExecuteReduceOnADevice<ReduceFuncColWise, T>(m_reduceFuncColWise, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, deviceID);
#endif

#ifdef USE_PINNED_MEMORY
      copyDeviceToHost(&result, d_output, 1, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]) );
#else
      copyDeviceToHost(&result, d_output, 1);
#endif

      cutilDeviceSynchronize();
   }
   else // do final reduction step on CPU instead
   {
      result = tempResult[0];

      for(size_t r=1; r<rows; r++)
      {
         result = m_reduceFuncColWise->CPU(result, tempResult[r]);
      }
   }

   freeCudaMemory<T>(deviceMemPointer);

   return result;
}





/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  matrix by using \em CUDA backend. Returns a scalar result. The function
 *  uses a variable number of devices, dividing the range of elemets equally
 *  among the participating devices each reducing its part. The results are
 *  then reduced themselves on the CPU.
 *
 *  \param input An input matrix whose elements need to be reduced.
 *  \param numDevices Integer deciding how many devices to utilize.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::reduceMultipleThreads_CU(SparseMatrix<T>& input, size_t numDevices)
{
   size_t rows = input.total_rows();
   size_t size = input.total_nnz();

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


      end = input.get_rowOffsetFromStart(numRows+i*numRowsPerSlice);

      in_mem_p[i] = input.updateDevice_CU((input.get_values()+offset), end-offset, i, false);

      size_t outSize=numRows*numBlocks;
      if(i==0 && outSize<(2*rows)) // for first device as later we may re-use this storage to do final reduction on GPU 0
         outSize = 2*rows; // make it at least that much large

      // Manually allocate output memory in this case
      allocateCudaMemory<T>(&deviceMemPointers[i], outSize);

      offset = end;
   }

   std::vector<T, malloc_allocator<T> > tempResult(rows);

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

      end = input.get_rowOffsetFromStart(numRows+i*numRowsPerSlice);

      in_mem_p[i] = input.updateDevice_CU((input.get_values()+offset), end-offset, i, true);

      d_input = in_mem_p[i]->getDeviceDataPointer();
      d_output = deviceMemPointers[i];

      // First reduce all elements row-wise so that each row produces one element.
      for(size_t r=0; r<numRows; r++)
      {
         size_t elemPerRow = input.get_rowSize(r+(numRowsPerSlice*i));

         if(elemPerRow>1)
         {
#ifdef USE_PINNED_MEMORY
            ExecuteReduceOnADevice<ReduceFuncRowWise, T>(m_reduceFuncRowWise, elemPerRow, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i, (device->m_streams[(r%maxKernelsSupported)]));
#else
            ExecuteReduceOnADevice<ReduceFuncRowWise, T>(m_reduceFuncRowWise, elemPerRow, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, i);
#endif

#ifdef USE_PINNED_MEMORY
            copyDeviceToHost(&tempResult[r+(numRowsPerSlice*i)], d_output, 1, (device->m_streams[(r%maxKernelsSupported)]) );
#else
            copyDeviceToHost(&tempResult[r+(numRowsPerSlice*i)], d_output, 1);
#endif
         }
         else
            tempResult[r+(numRowsPerSlice*i)] = ((elemPerRow>0) ? (input.begin(r+(numRowsPerSlice*i))(0)):T());  // dont use [] operator as that internally invalidate device copy

         d_input += elemPerRow;
         d_output += numBlocks;
      }

      offset = end;
   }

   // Synchronize all devices (?)
   m_environment->finishAll_CU(0, numDevices);

   T result;

   // if sufficient work then do final (column-wise) reduction on GPU
   if(rows>REDUCE_GPU_THRESHOLD)
   {
      cudaSetDevice(m_environment->bestCUDADevID); // do it on a single GPU or a CPU, should not be that much work(?)

      // reset to starting position and use it as an input
      d_input = deviceMemPointers[0];

      d_output = deviceMemPointers[0]+rows; // re-use already allocated space as well for output.

#ifdef USE_PINNED_MEMORY
      copyHostToDevice(&tempResult[0], d_input, rows, (m_environment->m_devices_CU.at(m_environment->bestCUDADevID)->m_streams[0]) );
#else
      copyHostToDevice(&tempResult[0], d_input, rows);
#endif

      getNumBlocksAndThreads(rows, maxBlocks, maxThreads, numBlocks, numThreads); // get numThreads and numBlocks for final reduction

#ifdef USE_PINNED_MEMORY
      ExecuteReduceOnADevice<ReduceFuncColWise, T>(m_reduceFuncColWise, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, m_environment->bestCUDADevID, (m_environment->m_devices_CU.at(m_environment->bestCUDADevID)->m_streams[0]));
#else
      ExecuteReduceOnADevice<ReduceFuncColWise, T>(m_reduceFuncColWise, rows, numThreads, numBlocks, maxThreads, maxBlocks, d_input, d_output, 0);
#endif

#ifdef USE_PINNED_MEMORY
      copyDeviceToHost(&result, d_output, 1, (m_environment->m_devices_CU.at(m_environment->bestCUDADevID)->m_streams[0]) );
#else
      copyDeviceToHost(&result, d_output, 1);
#endif

      cutilDeviceSynchronize();
   }
   else // do final reduction step on CPU instead
   {
      result = tempResult[0];

      for(size_t r=1; r<rows; r++)
      {
         result = m_reduceFuncColWise->CPU(result, tempResult[r]);
      }
   }

   // Free allocated memory on all devices, some pathetic issue, gives error when try to clear all poiters except first one
   for(size_t i = 0; i < numDevices; ++i)
   {
      cudaSetDevice(i);
      freeCudaMemory<T>(deviceMemPointers[i]);
   }

   return result;
}






















/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on a
 *  input Sparse Matrix. Returns a scalar result. Using the \em CUDA
 *  as backend. The function can be applied by any number of CUDA
 *  devices, thus internally calling the \em reduceSingle_CL or
 *  \em reduceNumDevices_CL depending upon number of
 *  CUDA devices specified/available.
 *
 *  \param input An input sparse matrix whose elements need to be reduced.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the 2D reduction performed.
 *
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::CU(SparseMatrix<T>& input, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE 2D SparseMatrix CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;

   if(numDevices < 2)
      return reduceSingleThread_CU(input, (m_environment->bestCUDADevID));
   else
      return reduceMultipleThreads_CU(input, numDevices);
}


}

#endif

