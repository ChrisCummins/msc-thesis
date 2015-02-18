/*! \file scan_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the Scan skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>

#include "operator_type.h"

#include "device_mem_pointer_cu.h"
#include "device_cu.h"
#include "scan_kernels.h"

namespace skepu
{

/*!
 *  Scans a Vector using the same recursive algorithm as NVIDIA SDK. First the vector is scanned producing partial results for each block.
 *  Then the function is called recursively to scan these partial results, which in turn can produce partial results and so on.
 *  This continues until only one block with partial results is left. Used by multi-GPU CUDA implementation.
 *
 *  \param input Pointer to the device memory where the input vector resides.
 *  \param output Pointer to the device memory where the output vector resides.
 *  \param blockSums A Vector of device memory pointers where the partial results for each level is stored.
 *  \param numElements The number of elements to scan.
 *  \param level The current recursion level.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param device Pointer to the device that will be used for the scan.
 *  \param scanFunc The user function used in the scan.
 */
template <typename BinaryFunc, typename T>
static T scanLargeVectorRecursivelyM_CU(DeviceMemPointer_CU<T>* input, DeviceMemPointer_CU<T>* output, std::vector<DeviceMemPointer_CU<T>*>& blockSums, size_t numElements, unsigned int level, ScanType type, T init, Device_CU* device, BinaryFunc scanFunc)
{
   unsigned int deviceID = device->getDeviceID();
   size_t numThreads = device->getMaxThreads();
   size_t maxBlocks = device->getMaxBlocks();
   const size_t numElementsPerThread = 1;
   size_t numBlocks = std::min(numElements/(numThreads*numElementsPerThread) + (numElements%(numThreads*numElementsPerThread) == 0 ? 0:1), maxBlocks);
   size_t totalNumBlocks = numElements/(numThreads*numElementsPerThread) + (numElements%(numThreads*numElementsPerThread) == 0 ? 0:1);
   size_t sharedElementsPerBlock = numThreads * numElementsPerThread;

   size_t sharedMemSize = sizeof(T) * (sharedElementsPerBlock*2);
   size_t updateSharedMemSize = sizeof(T) * (sharedElementsPerBlock);

   unsigned int isInclusive;
   if(type == INCLUSIVE)
      isInclusive = 1;
   else
      isInclusive = (level == 0) ? 0 : 1;

   T ret = 0;

   DeviceMemPointer_CU<T> ret_mem_p(&ret, 1, Environment<int>::getInstance()->m_devices_CU.at(deviceID));

   if (numBlocks > 1)
   {
#ifdef USE_PINNED_MEMORY
      ScanKernel_CU<<< numBlocks, numThreads, sharedMemSize, (Environment<int>::getInstance()->m_devices_CU.at(deviceID)->m_streams[0]) >>>(scanFunc, input->getDeviceDataPointer(), output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), sharedElementsPerBlock, numElements);
#else
      ScanKernel_CU<<< numBlocks, numThreads, sharedMemSize >>>(scanFunc, input->getDeviceDataPointer(), output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), sharedElementsPerBlock, numElements);
#endif

      scanLargeVectorRecursivelyM_CU(blockSums[level], blockSums[level], blockSums, totalNumBlocks, level+1, type, init, device, scanFunc);

#ifdef USE_PINNED_MEMORY
      ScanUpdate_CU<<< numBlocks, numThreads, updateSharedMemSize, (Environment<int>::getInstance()->m_devices_CU.at(deviceID)->m_streams[0]) >>>(scanFunc, output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), isInclusive, init, numElements, ret_mem_p.getDeviceDataPointer());
#else
      ScanUpdate_CU<<< numBlocks, numThreads, updateSharedMemSize >>>(scanFunc, output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), isInclusive, init, numElements, ret_mem_p.getDeviceDataPointer());
#endif
   }
   else
   {
#ifdef USE_PINNED_MEMORY
      ScanKernel_CU<<< numBlocks, numThreads, sharedMemSize, (Environment<int>::getInstance()->m_devices_CU.at(deviceID)->m_streams[0]) >>>(scanFunc, input->getDeviceDataPointer(), output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), sharedElementsPerBlock, numElements);

      ScanUpdate_CU<<< numBlocks, numThreads, updateSharedMemSize, (Environment<int>::getInstance()->m_devices_CU.at(deviceID)->m_streams[0]) >>>(scanFunc, output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), isInclusive, init, numElements, ret_mem_p.getDeviceDataPointer());
#else
      ScanKernel_CU<<< numBlocks, numThreads, sharedMemSize >>>(scanFunc, input->getDeviceDataPointer(), output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), sharedElementsPerBlock, numElements);

      ScanUpdate_CU<<< numBlocks, numThreads, updateSharedMemSize >>>(scanFunc, output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), isInclusive, init, numElements, ret_mem_p.getDeviceDataPointer());
#endif
   }

   ret_mem_p.changeDeviceData();
   ret_mem_p.copyDeviceToHost();

   return ret;
}


/*!
 *  Scans a Vector using the same recursive algorithm as NVIDIA SDK. First the vector is scanned producing partial results for each block.
 *  Then the function is called recursively to scan these partial results, which in turn can produce partial results and so on.
 *  This continues until only one block with partial results is left.
 *
 *  \param input Pointer to the device memory where the input vector resides.
 *  \param output Pointer to the device memory where the output vector resides.
 *  \param blockSums A Vector of device memory pointers where the partial results for each level is stored.
 *  \param numElements The number of elements to scan.
 *  \param level The current recursion level.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param deviceID Integer deciding which device to utilize.
 */
template <typename ScanFunc>
template <typename T>
T Scan<ScanFunc>::scanLargeVectorRecursively_CU(DeviceMemPointer_CU<T>* input, DeviceMemPointer_CU<T>* output, std::vector<DeviceMemPointer_CU<T>*>& blockSums, size_t numElements, unsigned int level, ScanType type, T init, unsigned int deviceID)
{
   BackEndParams bp=m_execPlan->find_(numElements);
   size_t numThreads = bp.maxThreads;
   size_t maxBlocks = bp.maxBlocks;
   const size_t numElementsPerThread = 1;
   size_t numBlocks = std::min(numElements/(numThreads*numElementsPerThread) + (numElements%(numThreads*numElementsPerThread) == 0 ? 0:1), maxBlocks);
   size_t totalNumBlocks = numElements/(numThreads*numElementsPerThread) + (numElements%(numThreads*numElementsPerThread) == 0 ? 0:1);
   size_t sharedElementsPerBlock = numThreads * numElementsPerThread;

   size_t sharedMemSize = sizeof(T) * (sharedElementsPerBlock*2);
   size_t updateSharedMemSize = sizeof(T) * (sharedElementsPerBlock);

   unsigned int isInclusive;
   if(type == INCLUSIVE)
      isInclusive = 1;
   else
      isInclusive = (level == 0) ? 0 : 1;

   T ret = 0;

   // Return value used for multi-GPU
   DeviceMemPointer_CU<T> ret_mem_p(&ret, 1, m_environment->m_devices_CU.at(deviceID));

   if (numBlocks > 1)
   {
      ScanKernel_CU<<< numBlocks, numThreads, sharedMemSize >>>(*m_scanFunc, input->getDeviceDataPointer(), output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), sharedElementsPerBlock, numElements);

      scanLargeVectorRecursively_CU(blockSums[level], blockSums[level], blockSums, totalNumBlocks, level+1, type, init, deviceID);

      ScanUpdate_CU<<< numBlocks, numThreads, updateSharedMemSize >>>(*m_scanFunc, output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), isInclusive, init, numElements, ret_mem_p.getDeviceDataPointer());
   }
   else
   {
      ScanKernel_CU<<< numBlocks, numThreads, sharedMemSize >>>(*m_scanFunc, input->getDeviceDataPointer(), output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), sharedElementsPerBlock, numElements);

      ScanUpdate_CU<<< numBlocks, numThreads, updateSharedMemSize >>>(*m_scanFunc, output->getDeviceDataPointer(), blockSums[level]->getDeviceDataPointer(), isInclusive, init, numElements, ret_mem_p.getDeviceDataPointer());
   }

   return ret;
}

/*!
 *  Performs the Scan on an input range using \em CUDA with a separate output range. Used when scanning the array on
 *  one device using one host thread. Allocates space for intermediate results from each block, and then calls scanLargeVectorRecursively_CU.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param deviceID Integer deciding which device to utilize.
 */
template <typename ScanFunc>
template <typename InputIterator, typename OutputIterator>
void Scan<ScanFunc>::scanSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, unsigned int deviceID)
{
   // Setup parameters
   size_t numElements = inputEnd-inputBegin;
   size_t numBlocks;
   size_t numThreads = m_execPlan->maxThreads(numElements);
   const size_t numElementsPerThread = 1;

   std::vector<DeviceMemPointer_CU<typename InputIterator::value_type>*> blockSums;

   size_t numEl = numElements;

   do
   {
      numBlocks = numEl/(numThreads*numElementsPerThread) + (numEl%(numThreads*numElementsPerThread) == 0 ? 0:1);
      if (numBlocks >= 1)
      {
         blockSums.push_back(new DeviceMemPointer_CU<typename InputIterator::value_type>(NULL, numBlocks, m_environment->m_devices_CU.at(deviceID)));
      }
      numEl = numBlocks;
   }
   while (numEl > 1);

   typename InputIterator::device_pointer_type_cu in_mem_p = inputBegin.getParent().updateDevice_CU(inputBegin.getAddress(), numElements, deviceID, true, false);
   typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU(outputBegin.getAddress(), numElements, deviceID, false, true);

   scanLargeVectorRecursively_CU(in_mem_p, out_mem_p, blockSums, numElements, 0, type, init, deviceID);

   //Clean up
   for(size_t i = 0; i < blockSums.size(); ++i)
   {
      delete blockSums[i];
   }

   out_mem_p->changeDeviceData();
}

/*!
 *  Performs the Scan on a whole Vector using \em CUDA with the input as output. A wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, int useNumGPU).
 *
 *  \param input A vector which will be scanned. It will be overwritten with the result.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::CU(Vector<T>& input, ScanType type, T init, int useNumGPU)
{
   CU(input.begin(), input.end(), input.begin(), type, init, useNumGPU);
}

/*!
 *  Performs the Scan on an input range using \em CUDA with the input range also used as an output. A wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, int useNumGPU).
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename ScanFunc>
template <typename InputIterator>
void Scan<ScanFunc>::CU(InputIterator inputBegin, InputIterator inputEnd, ScanType type, typename InputIterator::value_type init, int useNumGPU)
{
   CU(inputBegin, inputEnd, inputBegin, type, init, useNumGPU);
}

/*!
 *  Performs the Scan on a whole Vector using \em CUDA with a separate Vector as output. The output Vector will
 *  be resized an overwritten. A wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, int useNumGPU).
 *
 *  \param input A vector which will be scanned.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::CU(Vector<T>& input, Vector<T>& output, ScanType type, T init, int useNumGPU)
{
   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.size());
   }

   CU(input.begin(), input.end(), output.begin(), type, init, useNumGPU);
}

/*!
 *  Performs the Scan on an input range using \em CUDA with a separate output range. The function decides whether to perform
 *  the scan on one device, calling scanSingleThread_CU or
 *  on multiple devices, dividing the work between multiple devices.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename ScanFunc>
template <typename InputIterator, typename OutputIterator>
void Scan<ScanFunc>::CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("SCAN CUDA\n")

   size_t n = inputEnd - inputBegin;

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      scanSingleThread_CU(inputBegin, inputEnd, outputBegin, type, init, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      Vector<typename InputIterator::value_type> deviceSums;
      typename InputIterator::value_type ret = 0;

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;

         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         size_t numBlocks;
         size_t numThreads = m_execPlan->maxThreads(numElem);
         const size_t numElementsPerThread = 1;

         std::vector<DeviceMemPointer_CU<typename InputIterator::value_type>*> blockSums;

         size_t numEl = numElem;

         do
         {
            numBlocks = numEl/(numThreads*numElementsPerThread) + (numEl%(numThreads*numElementsPerThread) == 0 ? 0:1);
            if (numBlocks >= 1)
            {
               blockSums.push_back(new DeviceMemPointer_CU<typename InputIterator::value_type>(NULL, numBlocks, m_environment->m_devices_CU.at(i)));
            }
            numEl = numBlocks;
         }
         while (numEl > 1);

         typename InputIterator::device_pointer_type_cu in_mem_p = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numElem, i, true, false);
         typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem, i, false, true);

         cudaSetDevice(i);
         ret = scanLargeVectorRecursivelyM_CU(in_mem_p, out_mem_p, blockSums, numElem, 0, type, init, m_environment->m_devices_CU.at(i), *m_scanFunc);

         deviceSums.push_back(ret);

         out_mem_p->changeDeviceData();

         //Clean up
         for(size_t i = 0; i < blockSums.size(); ++i)
         {
            delete blockSums[i];
         }
      }

      CPU(deviceSums, INCLUSIVE);

      for(size_t i = 1; i < numDevices; ++i)
      {
         size_t numElements;
         if(i == numDevices-1)
            numElements = numElemPerSlice+rest;
         else
            numElements = numElemPerSlice;

         skepu::BackEndParams bp=m_execPlan->find_(inputEnd-inputBegin);
         size_t numThreads = bp.maxThreads;
         size_t maxBlocks = bp.maxBlocks;

         size_t numBlocks = std::min(numElements/(numThreads) + (numElements%(numThreads) == 0 ? 0:1), maxBlocks);

         typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElements, i, true, true);
         

         cudaSetDevice(i);

#ifdef USE_PINNED_MEMORY
         ScanAdd_CU<<< numBlocks, numThreads, 0, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(*m_scanFunc, out_mem_p->getDeviceDataPointer(), deviceSums(i-1), numElements);
#else
         ScanAdd_CU<<< numBlocks, numThreads >>>(*m_scanFunc, out_mem_p->getDeviceDataPointer(), deviceSums(i-1), numElements);
#endif
      }

      cudaSetDevice(m_environment->bestCUDADevID);
   }
}

}

#endif

