/*! \file mapreduce_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the MapReduce skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>

#include "operator_type.h"

#include "reduce_kernels.h"
#include "mapreduce_kernels.h"
#include "device_mem_pointer_cu.h"
#include "device_cu.h"

namespace skepu
{

/*!
 *  Performs the Map on \em one range of elements and Reduce on the result with \em CUDA as backend. Returns a scalar result. The function
 *  uses only \em one device which is decided by a parameter.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type MapReduce<MapFunc, ReduceFunc>::mapReduceSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID)
{
   cudaSetDevice(deviceID);

   // Setup parameters
   size_t n = inputEnd-inputBegin;
   typename InputIterator::value_type result = 0;

   size_t maxThreads = 256;  // number of threads per block, taken from NVIDIA source
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   getNumBlocksAndThreads(n, maxBlocks, maxThreads, numBlocks, numThreads);

   // Copies the elements to the device
   typename InputIterator::device_pointer_type_cu in_mem_p = inputBegin.getParent().updateDevice_CU(inputBegin.getAddress(), n, deviceID, true, false);

   // Create the output memory
   DeviceMemPointer_CU<typename InputIterator::value_type> out_mem_p(&result, numBlocks, m_environment->m_devices_CU.at(deviceID));

   typename InputIterator::value_type *d_idata= in_mem_p->getDeviceDataPointer();
   typename InputIterator::value_type *d_odata= out_mem_p.getDeviceDataPointer();

   // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
   size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(typename InputIterator::value_type) : numThreads * sizeof(typename InputIterator::value_type);

#ifdef USE_PINNED_MEMORY
   MapReduceKernel1_CU<<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapFunc, *m_reduceFunc, d_idata, d_odata, n);
#else
   MapReduceKernel1_CU<<<numBlocks, numThreads, sharedMemSize>>>(*m_mapFunc, *m_reduceFunc, d_idata, d_odata, n);
#endif

   size_t s=numBlocks;
   while(s > 1)
   {
      size_t threads = 0, blocks = 0;
      getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

#ifdef USE_PINNED_MEMORY
      CallReduceKernel_WithStream<ReduceFunc, typename InputIterator::value_type>(m_reduceFunc, s, threads, blocks, d_odata, d_odata, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]));
#else
      CallReduceKernel<ReduceFunc, typename InputIterator::value_type>(m_reduceFunc, s, threads, blocks, d_odata, d_odata);
#endif

      s = (s + (threads*2-1)) / (threads*2);
   }

   //Copy back result
   out_mem_p.changeDeviceData();
   out_mem_p.copyDeviceToHost(1);

   return result;
}


/*!
 *  Performs the Map on \em one Vector and Reduce on the result. Returns a scalar result.
 *  A wrapper for CU(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU).
 *  Using \em CUDA as backend.
 *
 *  \param input A vector which the map and reduce will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CU(Vector<T>& input, int useNumGPU)
{
   return CU(input.begin(), input.end(), useNumGPU);
}


/*!
 *  Performs the Map on \em one Matrix and Reduce on the result. Returns a scalar result.
 *  A wrapper for CU(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU).
 *  Using \em CUDA as backend.
 *
 *  \param input A matrix which the map and reduce will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CU(Matrix<T>& input, int useNumGPU)
{
   return CU(input.begin(), input.end(), useNumGPU);
}

/*!
 *  Performs the Map on \em one range of elements and Reduce on the result. Returns a scalar result. The function decides whether to perform
 *  the map and reduce on one device, calling mapReduceSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID) or
 *  on multiple devices, starting one thread per device where each thread runs mapReduceThreadFunc1_CU(void* _args).
 *  Using \em CUDA as backend.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type MapReduce<MapFunc, ReduceFunc>::CU(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      return mapReduceSingleThread_CU(inputBegin, inputEnd, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t n = inputEnd - inputBegin;

      // Divide elements among participating devices
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename InputIterator::value_type result[MAX_GPU_DEVICES];
      typename InputIterator::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
      typename InputIterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // Setup parameters
      size_t maxThreads = 256;
      size_t maxBlocks = 64;

      size_t numThreads[MAX_GPU_DEVICES];
      size_t numBlocks[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
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

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         // Decide size of shared memory
         size_t sharedMemSize = (numThreads[i] <= 32) ? 2 * numThreads[i] * sizeof(typename InputIterator::value_type) : numThreads[i] * sizeof(typename InputIterator::value_type);

         // Copies the elements to the device
         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numElem, i, true, false);

         // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
#ifdef USE_PINNED_MEMORY
         MapReduceKernel1_CU<<<numBlocks[i], numThreads[i], sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapFunc, *m_reduceFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#else
         MapReduceKernel1_CU<<<numBlocks[i], numThreads[i], sharedMemSize>>>(*m_mapFunc, *m_reduceFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#endif

         size_t s=numBlocks[i];
         while(s > 1)
         {
            size_t threads = 0, blocks = 0;
            getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

#ifdef USE_PINNED_MEMORY
            CallReduceKernel_WithStream<ReduceFunc, typename InputIterator::value_type>(m_reduceFunc, s, threads, blocks, out_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), (m_environment->m_devices_CU.at(i)->m_streams[0]));
#else
            CallReduceKernel<ReduceFunc, typename InputIterator::value_type>(m_reduceFunc, s, threads, blocks, out_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer());
#endif

            s = (s + (threads*2-1)) / (threads*2);
         }

         //Copy back result
         out_mem_p[i]->changeDeviceData();
      }

      // Joins the threads and reduces the results on the CPU, yielding the total result.
      out_mem_p[0]->copyDeviceToHost(1);
      typename InputIterator::value_type totalResult = result[0];
      delete out_mem_p[0];

      for(size_t i = 1; i < numDevices; ++i)
      {
         out_mem_p[i]->copyDeviceToHost(1);
         totalResult = m_reduceFunc->CPU(totalResult, result[i]);

         delete out_mem_p[i];
      }

      cudaSetDevice(m_environment->bestCUDADevID);

      return totalResult;
   }
}

/*!
 *  Performs the Map on \em two ranges of elements and Reduce on the result with \em CUDA as backend. Returns a scalar result. The function
 *  uses only \em one device which is decided by a parameter.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::mapReduceSingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, unsigned int deviceID)
{
   // Setup parameters
   size_t n = input1End-input1Begin;
   typename Input1Iterator::value_type result = 0;

   size_t maxThreads = 256;  // number of threads per block, taken from NVIDIA source
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   getNumBlocksAndThreads(n, maxBlocks, maxThreads, numBlocks, numThreads);

   // Copies the elements to the device
   typename Input1Iterator::device_pointer_type_cu in1_mem_p = input1Begin.getParent().updateDevice_CU(input1Begin.getAddress(), n, deviceID, true, false);
   typename Input2Iterator::device_pointer_type_cu in2_mem_p = input2Begin.getParent().updateDevice_CU(input2Begin.getAddress(), n, deviceID, true, false);

   // Create the output memory
   DeviceMemPointer_CU<typename Input1Iterator::value_type> out_mem_p(&result, numBlocks, m_environment->m_devices_CU.at(deviceID));

   typename Input1Iterator::value_type *d_idata1= in1_mem_p->getDeviceDataPointer();
   typename Input1Iterator::value_type *d_idata2= in2_mem_p->getDeviceDataPointer();
   typename Input1Iterator::value_type *d_odata= out_mem_p.getDeviceDataPointer();


   // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
   size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(typename Input1Iterator::value_type) : numThreads * sizeof(typename Input1Iterator::value_type);

#ifdef USE_PINNED_MEMORY
   MapReduceKernel2_CU<<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapFunc, *m_reduceFunc, d_idata1, d_idata2, d_odata, n);
#else
   MapReduceKernel2_CU<<<numBlocks, numThreads, sharedMemSize>>>(*m_mapFunc, *m_reduceFunc, d_idata1, d_idata2, d_odata, n);
#endif

   size_t s=numBlocks;
   while(s > 1)
   {
      size_t threads = 0, blocks = 0;
      getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

#ifdef USE_PINNED_MEMORY
      CallReduceKernel_WithStream<ReduceFunc, typename Input1Iterator::value_type>(m_reduceFunc, s, threads, blocks, d_odata, d_odata, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]));
#else
      CallReduceKernel<ReduceFunc, typename Input1Iterator::value_type>(m_reduceFunc, s, threads, blocks, d_odata, d_odata);
#endif

      s = (s + (threads*2-1)) / (threads*2);
   }

   //Copy back result
   out_mem_p.changeDeviceData();
   out_mem_p.copyDeviceToHost(1);

   return result;
}



/*!
 *  Performs the Map on \em two Vectors and Reduce on the result. Returns a scalar result.
 *  A wrapper for CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, int useNumGPU).
 *  Using \em CUDA as backend.
 *
 *  \param input1 A Vector which the map and reduce will be performed on.
 *  \param input2 A Vector which the map and reduce will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CU(Vector<T>& input1, Vector<T>& input2, int useNumGPU)
{
   return CU(input1.begin(), input1.end(), input2.begin(), input2.end(), useNumGPU);
}


/*!
 *  Performs the Map on \em two Matrices and Reduce on the result. Returns a scalar result.
 *  A wrapper for CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, int useNumGPU).
 *  Using \em CUDA as backend.
 *
 *  \param input1 A matrix which the map and reduce will be performed on.
 *  \param input2 A matrix which the map and reduce will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CU(Matrix<T>& input1, Matrix<T>& input2, int useNumGPU)
{
   return CU(input1.begin(), input1.end(), input2.begin(), input2.end(), useNumGPU);
}

/*!
 *  Performs the Map on \em two ranges of elements and Reduce on the result. Returns a scalar result. The function decides whether to perform
 *  the map and reduce on one device, calling mapReduceSingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, unsigned int deviceID) or
 *  on multiple devices, starting one thread per device where each thread runs mapReduceThreadFunc2_CU(void* _args).
 *  Using \em CUDA as backend.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, int useNumGPU)
{
   size_t numDevices = m_environment->m_devices_CU.size();
   
   DEBUG_TEXT_LEVEL1("MAPREDUCE CUDA GPUs: " << useNumGPU << ", numDevices: " << numDevices << "\n")

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      return mapReduceSingleThread_CU(input1Begin, input1End, input2Begin, input2End, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t n = input1End - input1Begin;

      // Divide elements among participating devices
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename Input1Iterator::value_type result[MAX_GPU_DEVICES];

      typename Input1Iterator::device_pointer_type_cu in1_mem_p[MAX_GPU_DEVICES];
      typename Input2Iterator::device_pointer_type_cu in2_mem_p[MAX_GPU_DEVICES];

      typename Input1Iterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // Setup parameters
      size_t maxThreads = 256;
      size_t maxBlocks = 64;

      size_t numThreads[MAX_GPU_DEVICES];
      size_t numBlocks[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         numBlocks[i] = numThreads[i] = 0;

         getNumBlocksAndThreads(numElem, maxBlocks, maxThreads, numBlocks[i], numThreads[i]);

         in1_mem_p[i] = input1Begin.getParent().updateDevice_CU((input1Begin+i*numElemPerSlice).getAddress(), numElem, i, false, false);
         in2_mem_p[i] = input2Begin.getParent().updateDevice_CU((input2Begin+i*numElemPerSlice).getAddress(), numElem, i, false, false);

         // Create the output memory
         out_mem_p[i] = new DeviceMemPointer_CU<typename Input1Iterator::value_type>(&result[i], numBlocks[i], m_environment->m_devices_CU.at(i));
      }

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         // Decide size of shared memory
         size_t sharedMemSize = (numThreads[i] <= 32) ? 2 * numThreads[i] * sizeof(typename Input1Iterator::value_type) : numThreads[i] * sizeof(typename Input1Iterator::value_type);

         // Copies the elements to the device
         in1_mem_p[i] = input1Begin.getParent().updateDevice_CU((input1Begin+i*numElemPerSlice).getAddress(), numElem, i, true, false);
         in2_mem_p[i] = input2Begin.getParent().updateDevice_CU((input2Begin+i*numElemPerSlice).getAddress(), numElem, i, true, false);

         // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
#ifdef USE_PINNED_MEMORY
         MapReduceKernel2_CU<<<numBlocks[i], numThreads[i], sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapFunc, *m_reduceFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#else
         MapReduceKernel2_CU<<<numBlocks[i], numThreads[i], sharedMemSize>>>(*m_mapFunc, *m_reduceFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#endif

         size_t s=numBlocks[i];
         while(s > 1)
         {
            size_t threads = 0, blocks = 0;
            getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

#ifdef USE_PINNED_MEMORY
            CallReduceKernel_WithStream<ReduceFunc, typename Input1Iterator::value_type>(m_reduceFunc, s, threads, blocks, out_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), (m_environment->m_devices_CU.at(i)->m_streams[0]));
#else
            CallReduceKernel<ReduceFunc, typename Input1Iterator::value_type>(m_reduceFunc, s, threads, blocks, out_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer());
#endif

            s = (s + (threads*2-1)) / (threads*2);
         }

         //mark data changed
         out_mem_p[i]->changeDeviceData();
      }

      // Joins the threads and reduces the results on the CPU, yielding the total result.
      out_mem_p[0]->copyDeviceToHost(1);
      typename Input1Iterator::value_type totalResult = result[0];
      delete out_mem_p[0];

      for(size_t i = 1; i < numDevices; ++i)
      {
         out_mem_p[i]->copyDeviceToHost(1);
         totalResult = m_reduceFunc->CPU(totalResult, result[i]);

         delete out_mem_p[i];
      }

      cudaSetDevice(m_environment->bestCUDADevID);
      return totalResult;
   }
}

/*!
 *  Performs the Map on \em three ranges of elements and Reduce on the result with \em CUDA as backend. Returns a scalar result. The function
 *  uses only \em one device which is decided by a parameter.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param input3Begin An iterator to the first element in the third range.
 *  \param input3End An iterator to the last element of the third range.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::mapReduceSingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, unsigned int deviceID)
{
   // Setup parameters
   size_t n = input1End-input1Begin;
   typename Input1Iterator::value_type result = 0;

   size_t maxThreads = 256;  // number of threads per block, taken from NVIDIA source
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   getNumBlocksAndThreads(n, maxBlocks, maxThreads, numBlocks, numThreads);

   // Copies the elements to the device
   typename Input1Iterator::device_pointer_type_cu in1_mem_p = input1Begin.getParent().updateDevice_CU(input1Begin.getAddress(), n, deviceID, true, false);
   typename Input2Iterator::device_pointer_type_cu in2_mem_p = input2Begin.getParent().updateDevice_CU(input2Begin.getAddress(), n, deviceID, true, false);
   typename Input3Iterator::device_pointer_type_cu in3_mem_p = input3Begin.getParent().updateDevice_CU(input3Begin.getAddress(), n, deviceID, true, false);

   // Create the output memory
   DeviceMemPointer_CU<typename Input1Iterator::value_type> out_mem_p(&result, numBlocks, m_environment->m_devices_CU.at(deviceID));

   typename Input1Iterator::value_type *d_idata1= in1_mem_p->getDeviceDataPointer();
   typename Input2Iterator::value_type *d_idata2= in2_mem_p->getDeviceDataPointer();
   typename Input3Iterator::value_type *d_idata3= in3_mem_p->getDeviceDataPointer();
   typename Input1Iterator::value_type *d_odata= out_mem_p.getDeviceDataPointer();


   // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
   size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(typename Input1Iterator::value_type) : numThreads * sizeof(typename Input1Iterator::value_type);

#ifdef USE_PINNED_MEMORY
   MapReduceKernel3_CU<<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapFunc, *m_reduceFunc, d_idata1, d_idata2, d_idata3, d_odata, n);
#else
   MapReduceKernel3_CU<<<numBlocks, numThreads, sharedMemSize>>>(*m_mapFunc, *m_reduceFunc, d_idata1, d_idata2, d_idata3, d_odata, n);
#endif

   size_t s=numBlocks;
   while(s > 1)
   {
      size_t threads = 0, blocks = 0;
      getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

#ifdef USE_PINNED_MEMORY
      CallReduceKernel_WithStream<ReduceFunc, typename Input1Iterator::value_type>(m_reduceFunc, s, threads, blocks, d_odata, d_odata, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]));
#else
      CallReduceKernel<ReduceFunc, typename Input1Iterator::value_type>(m_reduceFunc, s, threads, blocks, d_odata, d_odata);
#endif

      s = (s + (threads*2-1)) / (threads*2);
   }

   //Copy back result
   out_mem_p.changeDeviceData();
   out_mem_p.copyDeviceToHost(1);

   return result;
}

/*!
 *  Performs the Map on \em three Vectors and Reduce on the result. Returns a scalar result.
 *  A wrapper for CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, int useNumGPU).
 *  Using \em CUDA as backend.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param input3 A vector which the mapping will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CU(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, int useNumGPU)
{
   return CU(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), useNumGPU);
}


/*!
 *  Performs the Map on \em three matrices and Reduce on the result. Returns a scalar result.
 *  A wrapper for CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, int useNumGPU).
 *  Using \em CUDA as backend.
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param input3 A matrix which the mapping will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, int useNumGPU)
{
   return CU(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), useNumGPU);
}


/*!
 *  Performs the Map on \em three ranges of elements and Reduce on the result. Returns a scalar result. The function decides whether to perform
 *  the map and reduce on one device, calling mapReduceSingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, unsigned int deviceID) or
 *  on multiple devices, starting one thread per device where each thread runs mapReduceThreadFunc3_CU(void* _args).
 *  Using \em CUDA as backend.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param input3Begin An iterator to the first element in the third range.
 *  \param input3End An iterator to the last element of the third range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      return mapReduceSingleThread_CU(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t n = input1End - input1Begin;

      // Divide elements among participating devices
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename Input1Iterator::value_type result[MAX_GPU_DEVICES];

      typename Input1Iterator::device_pointer_type_cu in1_mem_p[MAX_GPU_DEVICES];
      typename Input2Iterator::device_pointer_type_cu in2_mem_p[MAX_GPU_DEVICES];
      typename Input3Iterator::device_pointer_type_cu in3_mem_p[MAX_GPU_DEVICES];

      typename Input1Iterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // Setup parameters
      size_t maxThreads = 256;
      size_t maxBlocks = 64;

      size_t numThreads[MAX_GPU_DEVICES];
      size_t numBlocks[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         numBlocks[i] = numThreads[i] = 0;

         getNumBlocksAndThreads(numElem, maxBlocks, maxThreads, numBlocks[i], numThreads[i]);

         in1_mem_p[i] = input1Begin.getParent().updateDevice_CU((input1Begin+i*numElemPerSlice).getAddress(), numElem, i, false, false);
         in2_mem_p[i] = input2Begin.getParent().updateDevice_CU((input2Begin+i*numElemPerSlice).getAddress(), numElem, i, false, false);
         in3_mem_p[i] = input3Begin.getParent().updateDevice_CU((input3Begin+i*numElemPerSlice).getAddress(), numElem, i, false, false);

         // Create the output memory
         out_mem_p[i] = new DeviceMemPointer_CU<typename Input1Iterator::value_type>(&result[i], numBlocks[i], m_environment->m_devices_CU.at(i));
      }

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         // Decide size of shared memory
         size_t sharedMemSize = (numThreads[i] <= 32) ? 2 * numThreads[i] * sizeof(typename Input1Iterator::value_type) : numThreads[i] * sizeof(typename Input1Iterator::value_type);


         // Copies the elements to the device
         in1_mem_p[i] = input1Begin.getParent().updateDevice_CU((input1Begin+i*numElemPerSlice).getAddress(), numElem, i, true, false);
         in2_mem_p[i] = input2Begin.getParent().updateDevice_CU((input2Begin+i*numElemPerSlice).getAddress(), numElem, i, true, false);
         in3_mem_p[i] = input3Begin.getParent().updateDevice_CU((input3Begin+i*numElemPerSlice).getAddress(), numElem, i, true, false);

         // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
#ifdef USE_PINNED_MEMORY
         MapReduceKernel3_CU<<<numBlocks[i], numThreads[i], sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapFunc, *m_reduceFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), in3_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#else
         MapReduceKernel3_CU<<<numBlocks[i], numThreads[i], sharedMemSize>>>(*m_mapFunc, *m_reduceFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), in3_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#endif

         size_t s=numBlocks[i];
         while(s > 1)
         {
            size_t threads = 0, blocks = 0;
            getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

#ifdef USE_PINNED_MEMORY
            CallReduceKernel_WithStream<ReduceFunc, typename Input1Iterator::value_type>(m_reduceFunc, s, threads, blocks, out_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), (m_environment->m_devices_CU.at(i)->m_streams[0]));
#else
            CallReduceKernel<ReduceFunc, typename Input1Iterator::value_type>(m_reduceFunc, s, threads, blocks, out_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer());
#endif

            s = (s + (threads*2-1)) / (threads*2);
         }

         //mark data changed
         out_mem_p[i]->changeDeviceData();


      }

      // Joins the threads and reduces the results on the CPU, yielding the total result.
      out_mem_p[0]->copyDeviceToHost(1);
      typename Input1Iterator::value_type totalResult = result[0];
      delete out_mem_p[0];

      for(size_t i = 1; i < numDevices; ++i)
      {
         out_mem_p[i]->copyDeviceToHost(1);
         totalResult = m_reduceFunc->CPU(totalResult, result[i]);

         delete out_mem_p[i];
      }

      cudaSetDevice(m_environment->bestCUDADevID);
      return totalResult;
   }
}

}

#endif

