/*! \file map_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the Map skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>

#include "operator_type.h"
#include "debug.h"
#include "map_kernels.h"
#include "device_mem_pointer_cu.h"
#include "device_cu.h"

namespace skepu
{

/*!
 *  Applies the Map skeleton to \em one range of elements specified by iterators. Result is saved to a seperate output range.
 *  The calculations are performed by one host thread using \p one device with \em CUDA as backend.
 *
 *  The skeleton must have been created with a \em unary user function.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element in the range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param deviceID Integer specifying the which device to use.
 */
template <typename MapFunc>
template <typename InputIterator, typename OutputIterator>
void Map<MapFunc>::mapSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, unsigned int deviceID)
{
   CHECK_CUDA_ERROR(cudaSetDevice(deviceID));
   
//    std::cerr << "%%%% Map GPU_" << deviceID << ", best: " << m_environment->bestCUDADevID << "\n\n";

   // Setup parameters
   size_t n = inputEnd-inputBegin;
   BackEndParams bp=m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;

   numThreads = std::min(maxThreads, n);
   numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));

   // Copies the elements to the device
   typename InputIterator::device_pointer_type_cu in_mem_p = inputBegin.getParent().updateDevice_CU( inputBegin.getAddress(), n, deviceID, true, false);
   typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU( outputBegin.getAddress(), n, deviceID, false, true);

#ifdef TESTING_EXCL_EXEC  
   cudaExecTimer.start();
   MapKernelUnary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
   cudaDeviceSynchronize();
   cudaExecTimer.stop();
   
   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
   
   return;
#endif   
   // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
   MapKernelUnary_CU<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#else
   MapKernelUnary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#endif

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();

#ifdef TUNER_MODE
   cudaDeviceSynchronize();
#endif
}




/*!
 *  Performs the Map on \em one element range with \em  CUDA as backend. Seperate output range. The Map skeleton needs to
 *  be created with a \em unary user function. Decides whether to use one device and mapSingleThread_CU or multiple devices.
 *  In the case of several devices the input ranges is divided evenly
 *  among the devices.
 *
 *  \param inputBegin An iterator to the first element in the first range.
 *  \param inputEnd An iterator to the last element of the first range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename InputIterator, typename OutputIterator>
void Map<MapFunc>::CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapSingleThread_CU(inputBegin, inputEnd, outputBegin, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t n = inputEnd - inputBegin;
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename InputIterator::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
      typename OutputIterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numElem,  i, false, false);
         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem,  i, false, false);
      }

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         //Copy input now
         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numElem,  i, true, false);
         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem,  i, false, true, true);

         // Setup parameters
         BackEndParams bp=m_execPlan->find_(numElem);
         size_t maxBlocks = bp.maxBlocks;
         size_t maxThreads = bp.maxThreads;
         size_t numBlocks;
         size_t numThreads;

         numThreads = std::min(maxThreads, numElem);
         numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

         // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
         MapKernelUnary_CU<<<numBlocks,numThreads,0, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#else
         MapKernelUnary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#endif

         // Copy back result (synchronizes)
         out_mem_p[i]->changeDeviceData();
      }

      CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
      
      outputBegin.getParent().setValidFlag(false); // set parent copy to invalid...
   }
}

/*!
 *  Performs mapping on \em one vector with \em CUDA backend. Input is used as output. The function is a wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(Vector<T>& input, int useNumGPU)
{
   CU(input.begin(), input.end(), input.begin(), useNumGPU);
}


/*!
 *  Performs mapping on \em one vector with \em CUDA backend. Seperate output vector. The function is a wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(Vector<T>& input, Vector<T>& output, int useNumGPU)
{
   CU(input.begin(), input.end(), output.begin(), useNumGPU);
}



/*!
 *  Performs mapping on \em one matrix with \em CUDA backend. Input is used as output. The function is a wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input A matrix which the mapping will be performed on. It will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(Matrix<T>& input, int useNumGPU)
{
   CU(input.begin(), input.end(), input.begin(), useNumGPU);
}


/*!
 *  Performs mapping on \em one matrix with \em CUDA backend. Seperate output matrix. The function is a wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(Matrix<T>& input, Matrix<T>& output, int useNumGPU)
{
   CU(input.begin(), input.end(), output.begin(), useNumGPU);
}

/*!
 *  Performs the Map on \em one element range with \em CUDA as backend. Input is used as output. The Map skeleton needs to
 *  be created with a \em unary user function. The function is a wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename InputIterator>
void Map<MapFunc>::CU(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU)
{
   CU(inputBegin, inputEnd, inputBegin, useNumGPU);
}

/*!
 *  Applies the Map skeleton to \em two ranges of elements specified by iterators. Result is saved to a seperate output range.
 *  The calculations are performed by one host thread using \p one device with \em CUDA as backend.
 *
 *  The skeleton must have been created with a \em binary user function.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param deviceID Integer specifying the which device to use.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
void Map<MapFunc>::mapSingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, unsigned int deviceID)
{
   CHECK_CUDA_ERROR(cudaSetDevice(deviceID));

   // Setup parameters
   size_t n = input1End-input1Begin;
   BackEndParams bp=m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;

   numThreads = std::min(maxThreads, n);
   numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));

   typename Input1Iterator::device_pointer_type_cu in1_mem_p = input1Begin.getParent().updateDevice_CU(input1Begin.getAddress(), n, deviceID, true, false);
   typename Input2Iterator::device_pointer_type_cu in2_mem_p = input2Begin.getParent().updateDevice_CU(input2Begin.getAddress(), n, deviceID, true, false);
   typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU(outputBegin.getAddress(), n, deviceID, false, true);

   // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
   MapKernelBinary_CU<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#else
   MapKernelBinary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#endif

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();

#ifdef TUNER_MODE
   cudaDeviceSynchronize();
#endif
}


/*!
 *  Performs mapping on \em two vectors with \em CUDA as backend. Seperate output vector. The function is a wrapper for
 *  CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(Vector<T>& input1, Vector<T>& input2, Vector<T>& output, int useNumGPU)
{
   CU(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin(), useNumGPU);
}




/*!
 *  Performs mapping on \em two matrices with \em CUDA as backend. Seperate output matrix. The function is a wrapper for
 *  CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& output, int useNumGPU)
{
   CU(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin(), useNumGPU);
}

/*!
 *  Performs the Map on \em two element ranges with \em  CUDA as backend. Seperate output range. The Map skeleton needs to
 *  be created with a \em binary user function. Decides whether to use one device and mapSingleThread_CU or multiple devices.
 *  In the case of several devices the input ranges is divided evenly
 *  among the devices.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
void Map<MapFunc>::CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP CUDA\n")

   size_t n = input1End - input1Begin;

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapSingleThread_CU(input1Begin, input1End, input2Begin, input2End, outputBegin, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename Input1Iterator::device_pointer_type_cu in1_mem_p[MAX_GPU_DEVICES];
      typename Input2Iterator::device_pointer_type_cu in2_mem_p[MAX_GPU_DEVICES];
      typename OutputIterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in1_mem_p[i] = input1Begin.getParent().updateDevice_CU((input1Begin+i*numElemPerSlice).getAddress(), numElem,  i, false, false);
         in2_mem_p[i] = input2Begin.getParent().updateDevice_CU((input2Begin+i*numElemPerSlice).getAddress(), numElem,  i, false, false);
         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem,  i, false, false);
      }

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in1_mem_p[i] = input1Begin.getParent().updateDevice_CU((input1Begin+i*numElemPerSlice).getAddress(), numElem,  i, true, false);
         in2_mem_p[i] = input2Begin.getParent().updateDevice_CU((input2Begin+i*numElemPerSlice).getAddress(), numElem,  i, true, false);
         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem,  i, false, true, true);

         // Setup parameters
         BackEndParams bp=m_execPlan->find_(numElem);
         size_t maxBlocks = bp.maxBlocks;
         size_t maxThreads = bp.maxThreads;
         size_t numBlocks;
         size_t numThreads;

         numThreads = std::min(maxThreads, numElem);
         numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

         // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
         MapKernelBinary_CU<<<numBlocks,numThreads,0, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#else
         MapKernelBinary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#endif

         // Change device data
         out_mem_p[i]->changeDeviceData();
      }

      CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
      
      outputBegin.getParent().setValidFlag(false);
   }
}

/*!
 *  Applies the Map skeleton to \em three ranges of elements specified by iterators. Result is saved to a seperate output range.
 *  The calculations are performed by one host thread using \p one device with \em CUDA as backend.
 *
 *  The skeleton must have been created with a \em trinary user function.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param input3Begin An iterator to the first element in the third range.
 *  \param input3End An iterator to the last element of the third range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param deviceID Integer specifying the which device to use.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
void Map<MapFunc>::mapSingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, unsigned int deviceID)
{
   CHECK_CUDA_ERROR(cudaSetDevice(deviceID));

   // Setup parameters
   size_t n = input1End-input1Begin;
   BackEndParams bp=m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;

   numThreads = std::min(maxThreads, n);
   numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));

   // Copies the elements to the device
   typename Input1Iterator::device_pointer_type_cu in1_mem_p = input1Begin.getParent().updateDevice_CU(input1Begin.getAddress(), n, deviceID, true, false);
   typename Input2Iterator::device_pointer_type_cu in2_mem_p = input2Begin.getParent().updateDevice_CU(input2Begin.getAddress(), n, deviceID, true, false);
   typename Input3Iterator::device_pointer_type_cu in3_mem_p = input3Begin.getParent().updateDevice_CU(input3Begin.getAddress(), n, deviceID, true, false);
   typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU(outputBegin.getAddress(), n, deviceID, false, true);

   // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
   MapKernelTrinary_CU<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), in3_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#else
   MapKernelTrinary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), in3_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#endif

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();

#ifdef TUNER_MODE
   cudaDeviceSynchronize();
#endif
}


/*!
 *  Performs mapping on \em three vectors with \em CUDA as backend. Seperate output vector. The function is a wrapper for
 *  CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param input3 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, Vector<T>& output, int useNumGPU)
{
   CU(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), output.begin(), useNumGPU);
}



/*!
 *  Performs mapping on \em three matrices with \em CUDA as backend. Seperate output matrix. The function is a wrapper for
 *  CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param input3 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, Matrix<T>& output, int useNumGPU)
{
   CU(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), output.begin(), useNumGPU);
}



/*!
 *  Performs the Map on \em three vector element ranges with \em  CUDA as backend. Seperate output range. The Map skeleton needs to
 *  be created with a \em trinary user function. Decides whether to use one device and mapSingleThread_CU or multiple devices.
 *  In the case of several devices the input ranges is divided evenly
 *  among the devices.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param input3Begin An iterator to the first element in the third range.
 *  \param input3End An iterator to the last element of the third range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
void Map<MapFunc>::CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP CUDA\n")

   size_t n = input1End - input1Begin;

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapSingleThread_CU(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, outputBegin, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename Input1Iterator::device_pointer_type_cu in1_mem_p[MAX_GPU_DEVICES];
      typename Input2Iterator::device_pointer_type_cu in2_mem_p[MAX_GPU_DEVICES];
      typename Input3Iterator::device_pointer_type_cu in3_mem_p[MAX_GPU_DEVICES];
      typename OutputIterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in1_mem_p[i] = input1Begin.getParent().updateDevice_CU((input1Begin+i*numElemPerSlice).getAddress(), numElem,  i, false, false);
         in2_mem_p[i] = input2Begin.getParent().updateDevice_CU((input2Begin+i*numElemPerSlice).getAddress(), numElem,  i, false, false);
         in3_mem_p[i] = input3Begin.getParent().updateDevice_CU((input3Begin+i*numElemPerSlice).getAddress(), numElem,  i, false, false);
         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem,  i, false, false);
      }

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in1_mem_p[i] = input1Begin.getParent().updateDevice_CU((input1Begin+i*numElemPerSlice).getAddress(), numElem,  i, true, false);
         in2_mem_p[i] = input2Begin.getParent().updateDevice_CU((input2Begin+i*numElemPerSlice).getAddress(), numElem,  i, true, false);
         in3_mem_p[i] = input3Begin.getParent().updateDevice_CU((input3Begin+i*numElemPerSlice).getAddress(), numElem,  i, true, false);
         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem,  i, false, true, true);

         // Setup parameters
         BackEndParams bp=m_execPlan->find_(numElem);
         size_t maxBlocks = bp.maxBlocks;
         size_t maxThreads = bp.maxThreads;
         size_t numBlocks;
         size_t numThreads;

         numThreads = std::min(maxThreads, numElem);
         numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

         // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
         MapKernelTrinary_CU<<<numBlocks,numThreads,0, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), in3_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), n);
#else
         MapKernelTrinary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), in3_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), n);
#endif

         // Change device data
         out_mem_p[i]->changeDeviceData();
      }

      CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
      
      outputBegin.getParent().setValidFlag(false);
   }
}


}

#endif

