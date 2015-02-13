/*! \file mapoverlap_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the MapOverlap skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>

#include "operator_type.h"

#include "mapoverlap_kernels.h"
#include "device_mem_pointer_cu.h"
#include "device_cu.h"

namespace skepu
{

// #define CHECK_CUDA_ERROR(err) if(err != cudaSuccess){std::cerr<<"CUDA Error: "<<cudaGetErrorString(err)<<"\n";}


/*!
 *  Performs the MapOverlap on a whole Vector. With itself as output. A wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU). Using \em CUDA as backend.
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CU(Vector<T>& input, EdgePolicy poly, T pad, int useNumGPU)
{
   CU(input.begin(), input.end(), poly, pad, useNumGPU);
}

/*!
 *  Performs the MapOverlap on a range of elements. With the same range as output. Since a seperate output is needed, a
 *  temporary output vector is created and copied to the input vector at the end. This is rather inefficient and the
 *  two functions using a seperated output explicitly should be used instead. Using \em CUDA as backend.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename InputIterator>
void MapOverlap<MapOverlapFunc>::CU(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU)
{
   Vector<typename InputIterator::value_type> output;
   output.clear();
   output.resize(inputEnd-inputBegin);

   CU(inputBegin, inputEnd, output.begin(), poly, pad, useNumGPU);

   for(InputIterator it = output.begin(); inputBegin != inputEnd; ++inputBegin, ++it)
   {
      *inputBegin = *it;
   }
}


/*!
 *  Applies the MapOverlap skeleton to a range of elements specified by iterators. Result is saved to a seperate output range.
 *  The function uses only \em one device which is decided by a parameter. Using \em CUDA as backend.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param deviceID Integer deciding which device to utilize.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID)
{
   cudaSetDevice(deviceID);

   // Setup parameters
   size_t n = inputEnd-inputBegin;
   BackEndParams bp = m_execPlan->find_(n);
   size_t numBlocks;
   size_t numThreads;
   size_t overlap = m_mapOverlapFunc->overlap;

   // Constructs a wrap vector, which is used if cyclic edge policy is specified.
   InputIterator inputEndMinusOverlap = (inputEnd - overlap);
   std::vector<typename InputIterator::value_type> wrap(2*overlap);


   /*
   * Usman: Changed this to only allocate wrap buffer when it is CYCLIC overlap as for
   * large overlap values it was causing extra overhead for CYCLIC and DUPLICATE policies to transfer wrap buffer as they don't use it
   */
   DeviceMemPointer_CU<typename InputIterator::value_type> *wrap_mem_p = NULL;

   if(poly == CYCLIC)
   {
      // Just update here to get latest values back.
      inputBegin.getParent().updateHost();

      for(size_t i = 0; i < overlap; ++i)
      {
         wrap[i] = inputEndMinusOverlap(i);
         wrap[overlap+i] = inputBegin(i);
      }
      // Copy wrap vector to device only if is is cyclic policy as otherwise it is not used.
      wrap_mem_p = new DeviceMemPointer_CU<typename InputIterator::value_type>(&wrap[0], wrap.size(), m_environment->m_devices_CU.at(deviceID));
      wrap_mem_p->copyHostToDevice();
   }
   else // construct a very naive dummy
      wrap_mem_p = new DeviceMemPointer_CU<typename InputIterator::value_type>(NULL, 1, m_environment->m_devices_CU.at(deviceID));

   numThreads = std::min(bp.maxThreads, n);

//----- START: error check for possible high overlap value which can bloat the shared memory ------//

   if(numThreads<overlap)
   {
      SKEPU_ERROR("Overlap is higher than maximum threads available. MapOverlap Aborted!!!\n");
   }

   size_t maxShMem = ((m_environment->m_devices_CU.at(deviceID)->getSharedMemPerBlock())/sizeof(typename InputIterator::value_type))-SHMEM_SAFITY_BUFFER; // little buffer for other usage

   size_t orgThreads = numThreads;

   numThreads = ( ((numThreads+2*overlap)<maxShMem)? numThreads : maxShMem-(2*overlap) );

   if(orgThreads != numThreads && (numThreads<8 || numThreads<overlap)) // if changed then atleast 8 threads should be there
   {
      SKEPU_ERROR("Error: Possibly overlap is too high for operation to be successful on this GPU. MapOverlap Aborted!!!\n");
   }

//----- END: error check for possible high overlap value which can bloat the shared memory ------//

   numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), bp.maxBlocks));
   size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

   // Copy elements to device and allocate output memory.
   typename InputIterator::device_pointer_type_cu in_mem_p = inputBegin.getParent().updateDevice_CU(inputBegin.getAddress(), n, deviceID, true, false);
   typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU(outputBegin.getAddress(), n, deviceID, false, true);

   // Launches the kernel (asynchronous), kernel is templetized and which version is chosen at compile time.
   if(poly == CONSTANT)
   {
#ifdef USE_PINNED_MEMORY
      MapOverlapKernel_CU<0><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p->getDeviceDataPointer(), n, 0, n, pad);
#else
      MapOverlapKernel_CU<0><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p->getDeviceDataPointer(), n, 0, n, pad);
#endif
   }
   else if(poly == CYCLIC)
   {
#ifdef USE_PINNED_MEMORY
      MapOverlapKernel_CU<1><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p->getDeviceDataPointer(), n, 0, n, pad);
#else
      MapOverlapKernel_CU<1><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p->getDeviceDataPointer(), n, 0, n, pad);
#endif
   }
   else if(poly == DUPLICATE)
   {
#ifdef USE_PINNED_MEMORY
      MapOverlapKernel_CU<2><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p->getDeviceDataPointer(), n, 0, n, pad);
#else
      MapOverlapKernel_CU<2><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p->getDeviceDataPointer(), n, 0, n, pad);
#endif
   }

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();

#ifdef TUNER_MODE
   cudaDeviceSynchronize();
#endif

   delete wrap_mem_p;
}

/*!
 *  Performs the MapOverlap on a whole Vector. With a seperate output vector. Wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU).
 *  Using \em CUDA as backend.
 *
 *  \param input A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CU(Vector<T>& input, Vector<T>& output, EdgePolicy poly, T pad, int useNumGPU)
{
   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.size());
   }

   CU(input.begin(), input.end(), output.begin(), poly, pad, useNumGPU);
}

/*!
 *  Performs the MapOverlap on a range of elements. With a seperate output range. Decides whether to use one device and mapOverlapSingleThread_CU or multiple devices
 *  and create new threads which calls mapOverlapThreadFunc_CU. In the case of several devices the input range is divided evenly
 *  among the threads created. Using \em CUDA as backend.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPOVERLAP CUDA\n")

   size_t totalNumElements = inputEnd - inputBegin;
   size_t overlap = m_mapOverlapFunc->overlap;

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapOverlapSingleThread_CU(inputBegin, inputEnd, outputBegin, poly, pad, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t numElemPerSlice = totalNumElements / numDevices;
      size_t rest = totalNumElements % numDevices;

      //Need to get new values from other devices so that the overlap between devices is up to date.
      //Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
//          inputBegin.getParent().updateHostAndInvalidateDevice();

      InputIterator inputEndMinusOverlap = (inputEnd - overlap);
      std::vector<typename InputIterator::value_type> wrap(2*overlap);
      if(poly == CYCLIC)
      {
         // Just update here to get latest values back.
         inputBegin.getParent().updateHost();

         for(size_t i = 0; i < overlap; ++i)
         {
            wrap[i] = inputEndMinusOverlap(i);
            wrap[overlap+i] = inputBegin(i);
         }
      }

      typename InputIterator::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
      typename OutputIterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];
      typename InputIterator::device_pointer_type_cu wrap_mem_p[MAX_GPU_DEVICES];

      size_t numBlocks[MAX_GPU_DEVICES];
      size_t numThreads[MAX_GPU_DEVICES];
      size_t n[MAX_GPU_DEVICES];

      size_t overlap = m_mapOverlapFunc->overlap;

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         n[i] = numElem;
         BackEndParams bp = m_execPlan->find_(n[i]);

         if(i == 0)
         {
            in_mem_p[i] = inputBegin.getParent().updateDevice_CU(inputBegin.getAddress(), numElem+overlap, i, false, false);
            n[i] = numElem+overlap;
         }
         else if(i == numDevices-1)
         {
            in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice-overlap).getAddress(), numElem+overlap, i, false, false);
            n[i] = numElem+overlap;
         }
         else
         {
            in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice-overlap).getAddress(), numElem+2*overlap, i, false, false);
            n[i] = numElem+2*overlap;
         }

         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem, i, false, false);

         // wrap vector, don't copy just allocates space onf CUDA device.
         wrap_mem_p[i] = new DeviceMemPointer_CU<typename InputIterator::value_type>(&wrap[0], wrap.size(), m_environment->m_devices_CU.at(i));

         numThreads[i] = std::min(bp.maxThreads, n[i]);

         //----- START: error check for possible high overlap value which can bloat the shared memory ------//

         if(numThreads[i]<overlap)
         {
            SKEPU_ERROR("Overlap is higher than maximum threads available. MapOverlap Aborted!!!\n");
         }

         size_t maxShMem = ((m_environment->m_devices_CU.at(i)->getSharedMemPerBlock())/sizeof(typename InputIterator::value_type))-SHMEM_SAFITY_BUFFER; // little buffer for other usage

         size_t orgThreads = numThreads[i];

         numThreads[i] = ( ((numThreads[i]+2*overlap)<maxShMem)? numThreads[i] : maxShMem-(2*overlap) );

         if(orgThreads != numThreads[i] && (numThreads[i]<8 || numThreads[i]<overlap)) // if changed then atleast 8 threads should be there
         {
            SKEPU_ERROR("Possibly overlap is too high for operation to be successful on this GPU. MapOverlap Aborted!!!\n");
         }

         //----- END: error check for possible high overlap value which can bloat the shared memory ------//

         numBlocks[i] = std::max((size_t)1, std::min( (n[i]/numThreads[i] + (n[i]%numThreads[i] == 0 ? 0:1)), bp.maxBlocks));
      }

      // parameters
      size_t out_offset, out_numelements;

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         if(poly == CYCLIC)
         {
            // Copy actual wrap vector to device only if it is cyclic policy. Otherwise it is just an extra overhead.
            wrap_mem_p[i]->copyHostToDevice(); // it take the main execution time
         }

         // Copy elemets to device and set other kernel parameters depending on which device it is, first, last or a middle device.
         if(i == 0)
         {
            in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numElem+overlap, i, true, false);
            out_offset = 0;
            out_numelements = numElem;
         }
         else if(i == numDevices-1)
         {
            in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice-overlap).getAddress(), numElem+overlap, i, true, false);
            out_offset = overlap;
            out_numelements = numElem;
         }
         else
         {
            in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice-overlap).getAddress(), numElem+2*overlap, i, true, false);
            out_offset = overlap;
            out_numelements = numElem;
         }
         
         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem, i, false, true, true);

         size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads[i]+2*overlap);

         // Launches the kernel (asynchronous), kernel is templetized and which version is chosen at compile time.
         if(poly == CONSTANT)
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU<0><<<numBlocks[i], numThreads[i], sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), n[i], out_offset, out_numelements, pad);
#else
            MapOverlapKernel_CU<0><<<numBlocks[i], numThreads[i], sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), n[i], out_offset, out_numelements, pad);
#endif
         }
         else if(poly == CYCLIC)
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU<1><<<numBlocks[i], numThreads[i], sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), n[i], out_offset, out_numelements, pad);
#else
            MapOverlapKernel_CU<1><<<numBlocks[i], numThreads[i], sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), n[i], out_offset, out_numelements, pad);
#endif
         }
         else if(poly == DUPLICATE)
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU<2><<<numBlocks[i], numThreads[i], sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), n[i], out_offset, out_numelements, pad);
#else
            MapOverlapKernel_CU<2><<<numBlocks[i], numThreads[i], sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), n[i], out_offset, out_numelements, pad);
#endif
         }

         // Make sure the data is marked as changed by the device
         out_mem_p[i]->changeDeviceData();
      }

      // to properly de-allocate the memory
      for(size_t i = 0; i < numDevices; ++i)
      {
         delete wrap_mem_p[i];
      }

      cudaSetDevice(m_environment->bestCUDADevID);

   } // end else
}



/*!
 * For Matrix overlap, we need to check whether overlap configuration is runnable considering total size of shared memory available on that system.
 * This method is a helper funtion doing that. It is called by another helper \p getThreadNumber_CU() method.
 *
 * \param numThreads Number of threads in a thread block.
 * \param deviceID The device ID.
 */
template <typename MapOverlapFunc>
template <typename T>
bool MapOverlap<MapOverlapFunc>::sharedMemAvailable_CU(size_t &numThreads, unsigned int deviceID)
{
   size_t overlap = m_mapOverlapFunc->overlap;

   size_t maxShMem = ((m_environment->m_devices_CU.at(deviceID)->getSharedMemPerBlock())/sizeof(T))-SHMEM_SAFITY_BUFFER; // little buffer for other usage

   size_t orgThreads = numThreads;

   numThreads = ( ((numThreads+2*overlap)<maxShMem)? numThreads : maxShMem-(2*overlap) );

   if(orgThreads == numThreads) // return true when nothing changed because of overlap constraint
      return true;

   if(numThreads<8) // not original numThreads then atleast 8 threads should be there
   {
      SKEPU_ERROR("Possibly overlap is too high for operation to be successful on this GPU. MapOverlap Aborted!!!\n");
      numThreads = 0;
   }
   return false;
}



/*!
 * Helper method used for calculating optimal thread count. For row- or column-wise overlap,
 * it determines a thread block size with perfect division of problem size.
 *
 * \param width The problem size.
 * \param numThreads Number of threads in a thread block.
 * \param deviceID The device ID.
 */
template <typename MapOverlapFunc>
template <typename T>
size_t MapOverlap<MapOverlapFunc>::getThreadNumber_CU(size_t width, size_t &numThreads, unsigned int deviceID)
{
   // first check whether shared memory would be ok for this numThreads. Changes numThreads accordingly
   if(!sharedMemAvailable_CU<T>(numThreads, deviceID) && numThreads<1)
   {
      std::cerr<<"Too lower overlap size to continue.\n";
      return 0;
   }

   if( (width % numThreads) == 0)
      return (width / numThreads);

   for(size_t i=numThreads-1; i>=1; i--) // decreament numThreads and see which one is a perfect divisor
   {
      if( (width % numThreads)==0)
         return (width / numThreads);
   }

   return 0;
}

/*!
 * Used for applying MapOverlap operation on a single CUDA device. Internally calls \em mapOverlapSingleThread_CU_Row or
 * \em mapOverlapSingleThread_CU_Col depending upon the overlap Policy specified.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param deviceID The integer specifying CUDA device to execute the operation.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
*/
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID,OverlapPolicy overlapPolicy)
{

   if(overlapPolicy == OVERLAP_ROW_COL_WISE)
   {
      Matrix<typename InputIterator::value_type> tmp_m(outputBegin.getParent().total_rows(), outputBegin.getParent().total_cols());
      mapOverlapSingleThread_CU_Row(inputBegin, inputEnd, tmp_m.begin(), poly, pad, deviceID);
      mapOverlapSingleThread_CU_Col(tmp_m.begin(),tmp_m.end(),outputBegin, poly, pad, deviceID);
   }
   else if(overlapPolicy == OVERLAP_COL_WISE)
      mapOverlapSingleThread_CU_Col(inputBegin, inputEnd, outputBegin, poly, pad, deviceID);
   else
      mapOverlapSingleThread_CU_Row(inputBegin, inputEnd, outputBegin, poly, pad, deviceID);
}


/*!
 *  Performs the column-wise MapOverlap on a range of elements, using 1 GPU, on the \em CUDA with a seperate output range.
 *  Used internally by other methods to apply column-wise mapoverlap operation.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param deviceID The integer specifying CUDA device to execute the operation.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapSingleThread_CU_Col(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID)
{
   cudaSetDevice(deviceID);

   size_t n = inputEnd-inputBegin;
   BackEndParams bp = m_execPlan->find_(n);

   size_t numBlocks;
   size_t numThreads = bp.maxThreads;

   size_t overlap = m_mapOverlapFunc->overlap;

   size_t colWidth = inputBegin.getParent().total_rows();
   size_t numCols = inputBegin.getParent().total_cols();

   size_t trdsize= colWidth;
   size_t blocksPerCol = 1;

   if(colWidth> numThreads)
   {
      size_t tmp= getThreadNumber_CU<typename InputIterator::value_type>(colWidth, numThreads, deviceID);

      if(tmp<1 || (tmp*numCols)>bp.maxBlocks)
      {
         SKEPU_ERROR("ERROR! Operation is larger than maximum block size! colWidth: "<<colWidth<<", numThreads: "<<numThreads<<" \n");
      }
      blocksPerCol = tmp;
      trdsize = colWidth / blocksPerCol;

      if(trdsize<overlap)
      {
         SKEPU_ERROR("ERROR! Cannot execute overlap with current overlap width.\n");
      }
   }

   InputIterator inputEndTemp = inputEnd;
   InputIterator inputBeginTemp = inputBegin;
   InputIterator inputEndMinusOverlap = (inputEnd - overlap);

   size_t twoOverlap=2*overlap;
   std::vector<typename InputIterator::value_type> wrap(twoOverlap*numCols);

   // Allocate wrap vector to device.
   DeviceMemPointer_CU<typename InputIterator::value_type> wrap_mem_p(&wrap[0], wrap.size(), m_environment->m_devices_CU.at(deviceID));

   if(poly == CYCLIC)
   {
      size_t stride = numCols;

      // Just update here to get latest values back.
      inputBegin.getParent().updateHost();
      for(size_t col=0; col< numCols; col++)
      {
         inputEndTemp = inputBeginTemp+(numCols*(colWidth-1));
         inputEndMinusOverlap = (inputEndTemp - (overlap-1)*stride);

         for(size_t i = 0; i < overlap; ++i)
         {
            wrap[i+(col*twoOverlap)] = inputEndMinusOverlap(i*stride);
            wrap[(overlap+i)+(col*twoOverlap)] = inputBeginTemp(i*stride);
         }
         inputBeginTemp++;
      }
   }
   if(poly == CYCLIC)
   {
      // Copy wrap vector only if it is a CYCLIC overlap policy.
      wrap_mem_p.copyHostToDevice();
   }

   numThreads = trdsize; //std::min(maxThreads, rowWidth);
   numBlocks = std::max((size_t)1, std::min( (blocksPerCol * numCols), bp.maxBlocks));
   size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

   // Copy elements to device and allocate output memory.
   typename InputIterator::device_pointer_type_cu in_mem_p = inputBegin.getParent().updateDevice_CU(inputBegin.getAddress(), colWidth, numCols, deviceID, true, false, false);
   typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU(outputBegin.getAddress(), colWidth, numCols, deviceID, false, true, false);


   m_mapOverlapFunc->setStride(1);

   // Launches the kernel (asynchronous), kernel is templetized and which version is chosen at compile time.
   if(poly == CONSTANT)
   {
#ifdef USE_PINNED_MEMORY
      MapOverlapKernel_CU_Matrix_Col<0><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerCol, numCols, colWidth);
#else
      MapOverlapKernel_CU_Matrix_Col<0><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerCol, numCols, colWidth);
#endif

   }
   else if(poly == CYCLIC)
   {
#ifdef USE_PINNED_MEMORY
      MapOverlapKernel_CU_Matrix_Col<1><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerCol, numCols, colWidth);
#else
      MapOverlapKernel_CU_Matrix_Col<1><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerCol, numCols, colWidth);
#endif
   }
   else if(poly == DUPLICATE)
   {
#ifdef USE_PINNED_MEMORY
      MapOverlapKernel_CU_Matrix_Col<2><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerCol, numCols, colWidth);
#else
      MapOverlapKernel_CU_Matrix_Col<2><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerCol, numCols, colWidth);
#endif
   }

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}






/*!
 *  Performs the column-wise MapOverlap on a range of elements, using multiple GPUs, on the \em CUDA with a seperate output range.
 *  Used internally by other methods to apply column-wise mapoverlap operation.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param numDevices Integer deciding how many devices to utilize.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapMultiThread_CU_Col(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices)
{
   size_t totalElems = inputEnd-inputBegin;
   BackEndParams bp = m_execPlan->find_(totalElems/numDevices);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;
   size_t overlap = m_mapOverlapFunc->overlap;

   size_t colWidth = inputBegin.getParent().total_rows();
   size_t numCols = inputBegin.getParent().total_cols();

   size_t numRowsPerSlice = (colWidth / numDevices);
   size_t numElemPerSlice = numRowsPerSlice*numCols;
   size_t restRows = (colWidth % numDevices);

   InputIterator inputEndTemp = inputEnd;
   InputIterator inputBeginTemp = inputBegin;
   InputIterator inputEndMinusOverlap = (inputEnd - overlap);

   std::vector<typename InputIterator::value_type> wrapStartDev(overlap*numCols);
   std::vector<typename InputIterator::value_type> wrapEndDev(overlap*numCols);

   if(poly == CYCLIC)
   {
      size_t stride = numCols;

      // Just update here to get latest values back.
      inputBegin.getParent().updateHost();
      for(size_t col=0; col< numCols; col++)
      {
         inputEndTemp = inputBeginTemp+(numCols*(colWidth-1));
         inputEndMinusOverlap = (inputEndTemp - (overlap-1)*stride);

         for(size_t i = 0; i < overlap; ++i)
         {
            wrapStartDev[i+(col*overlap)] = inputEndMinusOverlap(i*stride);
            wrapEndDev[i+(col*overlap)] = inputBeginTemp(i*stride);
         }
         inputBeginTemp++;
      }
   }

   typename InputIterator::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
   typename OutputIterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

   typename InputIterator::device_pointer_type_cu wrapStartDev_mem_p[MAX_GPU_DEVICES];
   typename InputIterator::device_pointer_type_cu wrapEndDev_mem_p[MAX_GPU_DEVICES];

   size_t trdsize[MAX_GPU_DEVICES];
   size_t blocksPerCol[MAX_GPU_DEVICES];

   // First create CUDA memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numRows;

      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      trdsize[i] = numRows;
      blocksPerCol[i] = 1;

      if(numRows> maxThreads)
      {
         size_t tmp= getThreadNumber_CU<typename InputIterator::value_type>(numRows, maxThreads, i);
         if(tmp<1 || (tmp*numCols)>maxBlocks)
         {
            SKEPU_ERROR("ERROR! Col width is larger than maximum thread size!"<<colWidth<<"  "<<maxThreads<<" \n");
         }
         blocksPerCol[i] = tmp;
         trdsize[i] = numRows / blocksPerCol[i];

         if(trdsize[i]<overlap)
         {
            SKEPU_ERROR("ERROR! Cannot execute overlap with current overlap width.\n");
         }
      }

      size_t overlapElems = overlap*numCols;

      if(i == 0)
         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numRows+overlap, numCols, i, false, false, false);
      else if(i == numDevices-1)
         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice-overlapElems).getAddress(), numRows+overlap, numCols, i, false, false, false);
      else
         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice-overlapElems).getAddress(), numRows+2*overlap, numCols, i, false, false, false);

      out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numRows, numCols, i, false, false, false);

      // wrap vector, don't copy just allocates space on CUDA device.
      wrapStartDev_mem_p[i] = new DeviceMemPointer_CU<typename InputIterator::value_type>(&wrapStartDev[0], wrapStartDev.size(), m_environment->m_devices_CU.at(i));
      wrapEndDev_mem_p[i] = new DeviceMemPointer_CU<typename InputIterator::value_type>(&wrapEndDev[0], wrapEndDev.size(), m_environment->m_devices_CU.at(i));
   }

   // we will divide the computation row-wise... copy data and do operation
   for(size_t i=0; i< numDevices; i++)
   {
      size_t n;
      size_t numRows;
      size_t numElems;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      size_t in_offset, overlapElems = overlap*numCols;
      numElems = numRows * numCols;

      // Copy elemets to device and set other kernel parameters depending on which device it is, first, last or a middle device.
      if(i == 0)
      {
         if(poly == CYCLIC)
            wrapStartDev_mem_p[i]->copyHostToDevice(); // copy wrap as it will be used partially, may optimize further by placing upper and lower overlap in separate buffers for muultiple devices?

         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numRows+overlap, numCols, i, true, false, false);

         in_offset = 0;
         n = numElems; //numElem+overlap;
      }
      else if(i == numDevices-1)
      {
         if(poly == CYCLIC)
            wrapEndDev_mem_p[i]->copyHostToDevice(); // copy wrap as it will be used partially, may optimize further by placing upper and lower overlap in separate buffers for muultiple devices?

         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice-overlapElems).getAddress(), numRows+overlap, numCols, i, true, false, false);

         in_offset = overlapElems;
         n = numElems; //+overlap;
      }
      else
      {
         in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice-overlapElems).getAddress(), numRows+2*overlap, numCols, i, true, false, false);

         in_offset = overlapElems;
         n = numElems; //+2*overlap;
      }
      
      out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numRows, numCols, i, false, true, false, true);

      numThreads = trdsize[i]; //std::min(maxThreads, rowWidth);
      numBlocks = std::max((size_t)1, std::min( (blocksPerCol[i] * numCols), maxBlocks));
      size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

      m_mapOverlapFunc->setStride(1);

      cudaSetDevice(i);

      // Launches the kernel (asynchronous), kernel is templetized and which version is chosen at compile time.
      if(poly == CONSTANT)
      {
         if(i == 0)
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU_Matrix_ColMulti<0, -1><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#else
            MapOverlapKernel_CU_Matrix_ColMulti<0, -1><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#endif
         }
         else if(i == numDevices-1)
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU_Matrix_ColMulti<0, 1><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#else
            MapOverlapKernel_CU_Matrix_ColMulti<0, 1><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#endif
         }
         else
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU_Matrix_ColMulti<0, 0><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#else
            MapOverlapKernel_CU_Matrix_ColMulti<0, 0><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#endif
         }

      }
      else if(poly == CYCLIC)
      {
         if(i == 0)
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU_Matrix_ColMulti<1, -1><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#else
            MapOverlapKernel_CU_Matrix_ColMulti<1, -1><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#endif
         }
         else if(i == numDevices-1)
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU_Matrix_ColMulti<1, 1><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapEndDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#else
            MapOverlapKernel_CU_Matrix_ColMulti<1, 1><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapEndDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#endif
         }
         else
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU_Matrix_ColMulti<1, 0><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#else
            MapOverlapKernel_CU_Matrix_ColMulti<1, 0><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#endif
         }
      }
      else if(poly == DUPLICATE)
      {
         if(i == 0)
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU_Matrix_ColMulti<2, -1><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#else
            MapOverlapKernel_CU_Matrix_ColMulti<2, -1><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#endif
         }
         else if(i == numDevices-1)
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU_Matrix_ColMulti<2, 1><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#else
            MapOverlapKernel_CU_Matrix_ColMulti<2, 1><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#endif
         }
         else
         {
#ifdef USE_PINNED_MEMORY
            MapOverlapKernel_CU_Matrix_ColMulti<2, 0><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#else
            MapOverlapKernel_CU_Matrix_ColMulti<2, 0><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrapStartDev_mem_p[i]->getDeviceDataPointer(), n, in_offset, n, pad, blocksPerCol[i], numCols, numRows);
#endif
         }
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }

   // to properly de-allocate the memory
   for(size_t i = 0; i < numDevices; ++i)
   {
      delete wrapStartDev_mem_p[i];
      delete wrapEndDev_mem_p[i];
   }

   cudaSetDevice(m_environment->bestCUDADevID);
}




/*!
 *  Performs the row-wise MapOverlap on a range of elements, using 1 GPU, on the \em CUDA with a seperate output range.
 *  Used internally by other methods to apply row-wise mapoverlap operation.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param deviceID The integer specifying CUDA device to execute the operation.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapSingleThread_CU_Row(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID)
{
   size_t n = inputEnd-inputBegin;
   BackEndParams bp = m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = 256; //bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;
   size_t overlap = m_mapOverlapFunc->overlap;

   size_t numrows = inputBegin.getParent().total_rows();
   size_t rowWidth = inputBegin.getParent().total_cols();

   size_t trdsize= rowWidth;
   size_t blocksPerRow = 1;
   if(rowWidth> maxThreads)
   {
      size_t tmp= getThreadNumber_CU<typename InputIterator::value_type>(rowWidth, maxThreads, deviceID);
      if(tmp<1 || (tmp*numrows)>maxBlocks)
      {
         SKEPU_ERROR("Row width is larger than maximum thread size."<<rowWidth<<"  "<<maxThreads<<" \n");
      }
      blocksPerRow = tmp;
      trdsize = rowWidth / blocksPerRow;

      if(trdsize<overlap)
      {
         SKEPU_ERROR("Cannot execute overlap with current overlap width.\n");
      }
   }


   InputIterator inputEndTemp = inputEnd;
   InputIterator inputBeginTemp = inputBegin;

   size_t twoOverlap=2*overlap;
   std::vector<typename InputIterator::value_type> wrap(twoOverlap*numrows);

   // Allocate wrap vector to device.
   DeviceMemPointer_CU<typename InputIterator::value_type> wrap_mem_p(&wrap[0], wrap.size(), m_environment->m_devices_CU.at(deviceID));

   if(poly == CYCLIC)
   {
      // Just update here to get latest values back.
      inputBegin.getParent().updateHost();
      for(size_t row=0; row< numrows; row++)
      {
         inputEndTemp = inputBeginTemp+rowWidth;

         for(size_t i = 0; i < overlap; ++i)
         {
            wrap[i+(row*twoOverlap)] = inputEndTemp(i-overlap);// inputEndMinusOverlap(i);
            wrap[(overlap+i)+(row*twoOverlap)] = inputBeginTemp(i);
         }
         inputBeginTemp += rowWidth;
      }
      // Copy wrap vector only if it is a CYCLIC overlap policy.
      wrap_mem_p.copyHostToDevice();
   }

   numThreads = trdsize; //std::min(maxThreads, rowWidth);
   numBlocks = std::max((size_t)1, std::min( (blocksPerRow * numrows), maxBlocks));
   size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

   
   // Copy elements to device and allocate output memory.
   typename InputIterator::device_pointer_type_cu in_mem_p = inputBegin.getParent().updateDevice_CU(inputBegin.getAddress(), numrows, rowWidth, deviceID, true, false, false);
   typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU(outputBegin.getAddress(), numrows, rowWidth, deviceID, false, true, false);

   m_mapOverlapFunc->setStride(1);

   // Launches the kernel (asynchronous), kernel is templetized and which version is chosen at compile time.
   if(poly == CONSTANT)
   {
#ifdef USE_PINNED_MEMORY
      MapOverlapKernel_CU_Matrix_Row<0><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerRow, rowWidth);
#else
      MapOverlapKernel_CU_Matrix_Row<0><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerRow, rowWidth);
#endif
   }
   else if(poly == CYCLIC)
   {
#ifdef USE_PINNED_MEMORY
      MapOverlapKernel_CU_Matrix_Row<1><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerRow, rowWidth);
#else
      MapOverlapKernel_CU_Matrix_Row<1><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerRow, rowWidth);
#endif
   }
   else if(poly == DUPLICATE)
   {
#ifdef USE_PINNED_MEMORY
      MapOverlapKernel_CU_Matrix_Row<2><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerRow, rowWidth);
#else
      MapOverlapKernel_CU_Matrix_Row<2><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), wrap_mem_p.getDeviceDataPointer(), n, 0, n, pad, blocksPerRow, rowWidth);
#endif
   }

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}





/*!
 *  Performs the row-wise MapOverlap on a range of elements, using multiple GPUs, on the \em CUDA with a seperate output range.
 *  Used internally by other methods to apply row-wise mapoverlap operation.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param numDevices Integer deciding how many devices to utilize.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::mapOverlapMultiThread_CU_Row(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices)
{
   size_t totalElems = inputEnd-inputBegin;
   BackEndParams bp = m_execPlan->find_(totalElems/numDevices);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;
   size_t overlap = m_mapOverlapFunc->overlap;

   size_t numrows = inputBegin.getParent().total_rows();
   size_t rowWidth = inputBegin.getParent().total_cols();

   size_t trdsize= rowWidth;
   size_t blocksPerRow = 1;
   if(rowWidth> maxThreads)
   {
      size_t tmp= getThreadNumber_CU<typename InputIterator::value_type>(rowWidth, maxThreads, 0);
      if(tmp<1 || (tmp*numrows)>maxBlocks)
      {
         SKEPU_ERROR("Row width is larger than maximum thread size!"<<rowWidth<<"  "<<maxThreads<<" \n");
      }
      blocksPerRow = tmp;
      trdsize = rowWidth / blocksPerRow;

      if(trdsize<overlap)
      {
         SKEPU_ERROR("Cannot execute overlap with current overlap width.\n");
      }
   }

   size_t numRowsPerSlice = (numrows / numDevices);
   size_t numElemPerSlice = numRowsPerSlice*rowWidth;
   size_t restRows = (numrows % numDevices);

   //Need to get new values from other devices so that the overlap between devices is up to date.
   //Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
//    inputBegin.getParent().updateHostAndInvalidateDevice();

   size_t twoOverlap=2*overlap;
   std::vector<typename InputIterator::value_type> wrap(twoOverlap*numrows);

   InputIterator inputEndTemp = inputEnd;
   InputIterator inputBeginTemp = inputBegin;

   if(poly == CYCLIC)
   {
      // Just update here to get latest values back.
      inputBegin.getParent().updateHost();
      for(size_t row=0; row< numrows; row++)
      {
         inputEndTemp = inputBeginTemp+rowWidth;

         for(size_t i = 0; i < overlap; ++i)
         {
            wrap[i+(row*twoOverlap)] = inputEndTemp(i-overlap);// inputEndMinusOverlap(i);
            wrap[(overlap+i)+(row*twoOverlap)] = inputBeginTemp(i);
         }
         inputBeginTemp += rowWidth;
      }
   }

   numThreads = trdsize; //std::min(maxThreads, rowWidth);

   m_mapOverlapFunc->setStride(1);


   typename InputIterator::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
   typename OutputIterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

   typename InputIterator::device_pointer_type_cu wrap_mem_p[MAX_GPU_DEVICES];

   // First create CUDA memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numRows;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      // Copy wrap vector to device.
      wrap_mem_p[i] = new DeviceMemPointer_CU<typename InputIterator::value_type>(&wrap[(i*numRowsPerSlice*overlap*2)], (numRowsPerSlice*overlap*2), m_environment->m_devices_CU.at(i));

      // Copy elements to device and allocate output memory.
      in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numRows, rowWidth, i, false, false, false);
      out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numRows, rowWidth, i, false, false, false);
   }

   // we will divide the computation row-wise... copy data and do operation
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numRows;
      size_t numElems;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      numElems = numRows * rowWidth;

      if(poly == CYCLIC)
      {
         // Copy wrap vector only if it is a CYCLIC overlap policy.
         wrap_mem_p[i]->copyHostToDevice();
      }


      numBlocks = std::max((size_t)1, std::min( (blocksPerRow * numRows), maxBlocks));
      size_t sharedMemSize = sizeof(typename InputIterator::value_type) * (numThreads+2*overlap);

      // Copy elements to device and allocate output memory.
      in_mem_p[i] = inputBegin.getParent().updateDevice_CU((inputBegin+i*numElemPerSlice).getAddress(), numRows, rowWidth, i, true, false, false);
      out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numRows, rowWidth, i, false, true, false);

      cudaSetDevice(i);

      // Launches the kernel (asynchronous), kernel is templetized and which version is chosen at compile time.
      if(poly == CONSTANT)
      {
#ifdef USE_PINNED_MEMORY
         MapOverlapKernel_CU_Matrix_Row<0><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), numElems, 0, numElems, pad, blocksPerRow, rowWidth);
#else
         MapOverlapKernel_CU_Matrix_Row<0><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), numElems, 0, numElems, pad, blocksPerRow, rowWidth);
#endif
      }
      else if(poly == CYCLIC)
      {
#ifdef USE_PINNED_MEMORY
         MapOverlapKernel_CU_Matrix_Row<1><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), numElems, 0, numElems, pad, blocksPerRow, rowWidth);
#else
         MapOverlapKernel_CU_Matrix_Row<1><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), numElems, 0, numElems, pad, blocksPerRow, rowWidth);
#endif
      }
      else if(poly == DUPLICATE)
      {
#ifdef USE_PINNED_MEMORY
         MapOverlapKernel_CU_Matrix_Row<2><<<numBlocks, numThreads, sharedMemSize, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), numElems, 0, numElems, pad, blocksPerRow, rowWidth);
#else
         MapOverlapKernel_CU_Matrix_Row<2><<<numBlocks, numThreads, sharedMemSize>>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), wrap_mem_p[i]->getDeviceDataPointer(), numElems, 0, numElems, pad, blocksPerRow, rowWidth);
#endif
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }

   // to properly de-allocate the memory
   for(size_t i = 0; i < numDevices; ++i)
   {
      delete wrap_mem_p[i];
   }

   cudaSetDevice(m_environment->bestCUDADevID);
}

/*!
 *  Performs the MapOverlap on a whole Matrix. With a seperate output matrix. Wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename
 *  InputIterator::value_type pad, int useNumGPU).
 *  Using \em CUDA as backend.
 *
 *  \param input A Matrix which the mapping will be performed on.
 *  \param output The result Matrix, will be overwritten with the result and resized if needed.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CU(Matrix<T>& input, Matrix<T>& output, OverlapPolicy overlapPolicy, EdgePolicy poly, T pad, int useNumGPU)
{
   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.total_rows(), input.total_cols());
   }

   CU(input.begin(), input.end(), output.begin(), overlapPolicy, poly, pad, useNumGPU);
}

/*!
 *  Performs the MapOverlap on a range of elements. With a seperate output range. Decides whether to use
 *  one device or multiple devices
 *  In the case of several devices the input range is divided evenly
 *  among the multiple devices. Using \em CUDA as backend.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPOVERLAP CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapOverlapSingleThread_CU(inputBegin, inputEnd, outputBegin, poly, pad, (m_environment->bestCUDADevID), overlapPolicy);
   }
   else
   {
      if(overlapPolicy == OVERLAP_ROW_COL_WISE)
      {
         Matrix<typename InputIterator::value_type> tmp_m(outputBegin.getParent().total_rows(), outputBegin.getParent().total_cols());
         mapOverlapMultiThread_CU_Row(inputBegin, inputEnd, tmp_m.begin(), poly, pad, numDevices);
         mapOverlapMultiThread_CU_Col(tmp_m.begin(), tmp_m.end(), outputBegin, poly, pad, numDevices);
      }
      else if(overlapPolicy == OVERLAP_COL_WISE)
         mapOverlapMultiThread_CU_Col(inputBegin, inputEnd, outputBegin, poly, pad, numDevices);
      else
         mapOverlapMultiThread_CU_Row(inputBegin, inputEnd, outputBegin, poly, pad, numDevices);
   }
}

/*!
 *  Performs the MapOverlap on a whole Matrix. With itself as output. A wrapper for
 *  CU(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU). Using \em CUDA as backend.
 *
 *  \param input A Matrix which the mapping will be performed on. It will be overwritten with the result.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::CU(Matrix<T>& input, OverlapPolicy overlapPolicy, EdgePolicy poly, T pad, int useNumGPU)
{
   CU(input.begin(), input.end(), overlapPolicy, poly, pad, useNumGPU);
}

/*!
 *  Performs the MapOverlap on a range of elements. With the same range as output. Since a seperate output is needed, a
 *  temporary output matrix is created and copied to the input matrix at the end. This is rather inefficient and the
 *  two functions using a seperated output explicitly should be used instead. Using \em CUDA as backend.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlapFunc>
template <typename InputIterator>
void MapOverlap<MapOverlapFunc>::CU(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad, int useNumGPU)
{
   Matrix<typename InputIterator::value_type> output;
   output.clear();
   output.resize(inputBegin.getParent().total_rows(), inputBegin.getParent().total_cols());

   CU(inputBegin, inputEnd, output.begin(), overlapPolicy, poly, pad, useNumGPU);

   for(InputIterator it = output.begin(); inputBegin != inputEnd; ++inputBegin, ++it)
   {
      *inputBegin = *it;
   }
}




}
#endif


