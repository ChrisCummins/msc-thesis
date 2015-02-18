/*! \file generate_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the Generate skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>

#include "operator_type.h"

#include "generate_kernels.h"
#include "device_mem_pointer_cu.h"
#include "device_cu.h"

namespace skepu
{

/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output range. It generates the elements using \em CUDA as backend on \em one device.
 *
 *  \param numElements The number of elements to be generated.
 *  \param outputBegin An iterator pointing to the first element in the range which will be overwritten with generated values.
 *  \param deviceID Integer specifying the which device to use.
 */
template <typename GenerateFunc>
template <typename OutputIterator>
void Generate<GenerateFunc>::generateSingleThread_CU(size_t numElements, OutputIterator outputBegin, unsigned int deviceID)
{
   CHECK_CUDA_ERROR(cudaSetDevice(deviceID));
//    std::cerr << "%%%% Generating GPU_" << deviceID << ", best: " << m_environment->bestCUDADevID << "\n\n";

   // Setup parameters
   size_t n = numElements;
   BackEndParams bp=m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;

   numThreads = std::min(maxThreads, n);
   numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));

   // Copies the elements to the device
   typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU(outputBegin.getAddress(), n, deviceID, false, true);

   // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
   GenerateKernel_CU<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_generateFunc, out_mem_p->getDeviceDataPointer(), n, 0);
#else
   GenerateKernel_CU<<<numBlocks,numThreads>>>(*m_generateFunc, out_mem_p->getDeviceDataPointer(), n, 0);
#endif

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}

/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output vector which is resized to numElements. A wrapper for
 *  CU(size_t numElements, OutputIterator outputBegin, int useNumGPU). For the \em CUDA backend.
 *
 *  \param numElements The number of elements to be generated.
 *  \param output The output vector which will be overwritten with the generated values.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename GenerateFunc>
template <typename T>
void Generate<GenerateFunc>::CU(size_t numElements, Vector<T>& output, int useNumGPU)
{
   if(output.size() != numElements)
   {
      output.clear();
      output.resize(numElements);
   }

   CU(numElements, output.begin(), useNumGPU);
}



/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output matrix which is resized to numElements. For the \em CUDA backend.
 *
 *  \param numRows The number of rows to be generated.
 *  \param numCols The number of columns to be generated.
 *  \param output The output matrix which will be overwritten with the generated values.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename GenerateFunc>
template <typename T>
void Generate<GenerateFunc>::CU(size_t numRows, size_t numCols, Matrix<T>& output, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("GENERATE CUDA Matrix\n")

   if((output.total_rows() != numRows) && (output.total_cols() != numCols))
   {
      output.clear();
      output.resize(numRows, numCols);
   }

   size_t numDevices = m_environment->m_devices_CU.size();
   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }

   size_t numRowsPerSlice = numRows / numDevices;
   size_t restRows = numRows % numDevices;

   size_t numElemPerSlice = numRowsPerSlice*numCols;
   size_t restElem = restRows * numCols;

   typename Matrix<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

   // First create CUDA memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+restElem;
      else
         numElem = numElemPerSlice;

      out_mem_p[i] = output.updateDevice_CU((output.getAddress()+i*numElemPerSlice), numElem,  i, false, true, true);
   }

   size_t offsetInRows = 0;

   // Fill out argument struct with right information and start threads.
   for(size_t i = 0; i < numDevices; ++i)
   {
      CHECK_CUDA_ERROR(cudaSetDevice(i));

      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+restElem;
      else
         numElem = numElemPerSlice;

      size_t nrows = numElem/numCols;

      // Setup parameters
      dim3 numThreads, numBlocks;

      numThreads.x =  (numCols>32)? 32: numCols;                            // each thread does multiple Xs
      numThreads.y =  (nrows>16)? 16: nrows;
      numThreads.z = 1;
      numBlocks.x = (numCols+(numThreads.x-1)) / numThreads.x;
      numBlocks.y = (nrows+(numThreads.y-1))  / numThreads.y;
      numBlocks.z = 1;

      // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
      GenerateKernel_CU_Matrix<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(*m_generateFunc, out_mem_p[i]->getDeviceDataPointer(), numElem, numCols, nrows, offsetInRows);
#else
      GenerateKernel_CU_Matrix<<<numBlocks,numThreads>>>(*m_generateFunc, out_mem_p[i]->getDeviceDataPointer(), numElem, numCols, nrows, offsetInRows);
#endif

      out_mem_p[i]->changeDeviceData();

      offsetInRows += (numElem/numCols);
   }
   
   output.setValidFlag(false);

   CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
}

/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output range. For the \em CUDA backend.
 *  Decides whether to use one device and generateSingleThread_CU or multiple devices
 *  and create new threads which calls generateThreadFunc_CU. In the case of several devices the input range is divided evenly
 *  among the threads created.
 *
 *  \param numElements The number of elements to be generated.
 *  \param outputBegin An iterator pointing to the first element in the range which will be overwritten with generated values.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename GenerateFunc>
template <typename OutputIterator>
void Generate<GenerateFunc>::CU(size_t numElements, OutputIterator outputBegin, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("GENERATE CUDA\n")

   size_t n = numElements;

   size_t numDevices = m_environment->m_devices_CU.size();
   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      generateSingleThread_CU(numElements, outputBegin, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename OutputIterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem,  i, false, true);
      }

      size_t offset = 0;

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         size_t n = numElements;
         BackEndParams bp=m_execPlan->find_(numElem);
         size_t maxBlocks = bp.maxBlocks;
         size_t maxThreads = bp.maxThreads;
         size_t numBlocks;
         size_t numThreads;

         numThreads = std::min(maxThreads, n);
         numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));

         // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
         GenerateKernel_CU<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(*m_generateFunc, out_mem_p[i]->getDeviceDataPointer(), numElem, offset);
#else
         GenerateKernel_CU<<<numBlocks,numThreads>>>(*m_generateFunc, out_mem_p[i]->getDeviceDataPointer(), numElem, offset);
#endif

         out_mem_p[i]->changeDeviceData();

         offset += numElem;
      }

      CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
   }
}


}

#endif

