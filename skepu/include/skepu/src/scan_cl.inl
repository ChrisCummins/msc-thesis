/*! \file scan_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the Scan skeleton.
 */

#ifdef SKEPU_OPENCL

#include <iostream>

#include "operator_type.h"

#include "scan_kernels.h"

namespace skepu
{


/*!
 *  A function called by the constructor. It creates the OpenCL program for the skeleton and saves a handle for
 *  the kernel. The program is built from a string containing the user function (specified when constructing the
 *  skeleton) and a generic Scan kernel. The type and function names in the generic kernel are relpaced by user function
 *  specific code before it is compiled by the OpenCL JIT compiler. The Scan kernel actually is two kernels which both
 *  have their handles saved. The actual scan kernel and a uniform add kernel to add the block sums produced by scanning
 *
 *  Also handles the use of doubles automatically by including "#pragma OPENCL EXTENSION cl_khr_fp64: enable" if doubles
 *  are used.
 */
template <typename ScanFunc>
void Scan<ScanFunc>::createOpenCLProgram()
{
   //Creates the sourcecode
   std::string totalSource;
   std::string kernelSource;
   std::string funcSource = m_scanFunc->func_CL;
   std::string kernelName, updateName, addName;

   if(m_scanFunc->datatype_CL == "double")
   {
      totalSource.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");
   }

   kernelSource = ScanKernel_CL + ScanUpdate_CL + ScanAdd_CL;
   kernelName = "ScanKernel_" + m_scanFunc->funcName_CL;
   updateName = "ScanUpdate_" + m_scanFunc->funcName_CL;
   addName = "ScanAdd_" + m_scanFunc->funcName_CL;
   
   replaceTextInString(kernelSource, std::string("TYPE"), m_scanFunc->datatype_CL);
   replaceTextInString(kernelSource, std::string("KERNELNAME"), m_scanFunc->funcName_CL);
   replaceTextInString(kernelSource, std::string("FUNCTIONNAME"), m_scanFunc->funcName_CL);

   // check for extra user-supplied opencl code for custome datatype
   totalSource.append( read_file_into_string(OPENCL_SOURCE_FILE_NAME) );

   totalSource.append(funcSource);
   totalSource.append(kernelSource);
   
   if(sizeof(size_t) <= 4)
      replaceTextInString(totalSource, std::string("size_t "), "unsigned int ");
   else if(sizeof(size_t) <= 8)
      replaceTextInString(totalSource, std::string("size_t "), "unsigned long ");
   else
      SKEPU_ERROR("OpenCL code compilation issue: sizeof(size_t) is bigger than 8 bytes: " << sizeof(size_t));

   DEBUG_TEXT_LEVEL3(totalSource);

   //Builds the code and creates kernel for all devices
   for(std::vector<Device_CL*>::iterator it = m_environment->m_devices_CL.begin(); it != m_environment->m_devices_CL.end(); ++it)
   {
      const char* c_src = totalSource.c_str();
      cl_int err;
      cl_program temp_program;
      cl_kernel temp_kernel;

      temp_program = clCreateProgramWithSource((*it)->getContext(), 1, &c_src, NULL, &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating OpenCLsource!!\n");
      }

      err = clBuildProgram(temp_program, 0, NULL, NULL, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         #if SKEPU_DEBUG > 0
            cl_build_status build_status;
            clGetProgramBuildInfo(temp_program, (*it)->getDeviceID(), CL_PROGRAM_BUILD_STATUS, sizeof(cl_build_status), &build_status, NULL);
            if (build_status != CL_SUCCESS)
            {
               char *build_log;
               size_t ret_val_size;
               clGetProgramBuildInfo(temp_program, (*it)->getDeviceID(), CL_PROGRAM_BUILD_LOG, 0, NULL, &ret_val_size);
               build_log = new char[ret_val_size+1];
               clGetProgramBuildInfo(temp_program, (*it)->getDeviceID(), CL_PROGRAM_BUILD_LOG, ret_val_size, build_log, NULL);

               // to be carefully, terminate with \0
               // there's no information in the reference whether the string is 0 terminated or not
               build_log[ret_val_size] = '\0';

               SKEPU_ERROR("Error building OpenCL program!!\n" <<err <<"\nBUILD LOG:\n" << build_log);

               delete[] build_log;
            }
         #else
            SKEPU_ERROR("Error building OpenCL program. " << err <<"\n");
         #endif
      }

      temp_kernel = clCreateKernel(temp_program, kernelName.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<kernelName.c_str() <<"\n" <<err <<"\n");
      }

      m_scanKernels_CL.push_back(std::make_pair(temp_kernel, (*it)));

      temp_kernel = clCreateKernel(temp_program, updateName.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<updateName.c_str() <<"\n" <<err <<"\n");
      }

      m_scanUpdateKernels_CL.push_back(std::make_pair(temp_kernel, (*it)));

      temp_kernel = clCreateKernel(temp_program, addName.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<addName.c_str() <<"\n" <<err <<"\n");
      }

      m_scanAddKernels_CL.push_back(std::make_pair(temp_kernel, (*it)));
   }
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
T Scan<ScanFunc>::scanLargeVectorRecursively_CL(DeviceMemPointer_CL<T>* input, DeviceMemPointer_CL<T>* output, std::vector<DeviceMemPointer_CL<T>*>& blockSums, size_t numElements, unsigned int level, ScanType type, T init, unsigned int deviceID)
{
   cl_int err;

   size_t globalWorkSize[1];
   size_t localWorkSize[1];

   skepu::BackEndParams bp=m_execPlan->find_(numElements);
   size_t numThreads = bp.maxThreads;
   size_t maxBlocks = bp.maxBlocks;
   const size_t numElementsPerThread = 1;
   size_t numBlocks = std::min(numElements/(numThreads*numElementsPerThread) + (numElements%(numThreads*numElementsPerThread) == 0 ? 0:1), maxBlocks);
   size_t totalNumBlocks = numElements/(numThreads*numElementsPerThread) + (numElements%(numThreads*numElementsPerThread) == 0 ? 0:1);
   size_t sharedElementsPerBlock = numThreads * numElementsPerThread;

   size_t sharedMemSize = sizeof(T) * (sharedElementsPerBlock*2);
   size_t updateSharedMemSize = sizeof(T) * (sharedElementsPerBlock);

   int isInclusive;
   if(type == INCLUSIVE)
      isInclusive = 1;
   else
      isInclusive = (level == 0) ? 0 : 1;

   T tempInit = init;
   T ret = 0;

   DeviceMemPointer_CL<T> ret_mem_p(&ret, 1, m_scanKernels_CL.at(deviceID).second);

   cl_kernel scanKernel = m_scanKernels_CL.at(deviceID).first;
   cl_kernel scanUpdate = m_scanUpdateKernels_CL.at(deviceID).first;

   cl_mem in_p = input->getDeviceDataPointer();
   cl_mem out_p = output->getDeviceDataPointer();
   cl_mem blockSums_p = blockSums[level]->getDeviceDataPointer();
   cl_mem ret_p = ret_mem_p.getDeviceDataPointer();

   if (numBlocks > 1)
   {
      clSetKernelArg(scanKernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(scanKernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(scanKernel, 2, sizeof(cl_mem), (void*)&blockSums_p);
      clSetKernelArg(scanKernel, 3, sizeof(size_t), (void*)&sharedElementsPerBlock);
      clSetKernelArg(scanKernel, 4, sizeof(size_t), (void*)&numElements);
      clSetKernelArg(scanKernel, 5, sharedMemSize, NULL);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      err = clEnqueueNDRangeKernel(m_scanKernels_CL.at(deviceID).second->getQueue(), scanKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching Scan kernel!! " <<err <<"\n");
      }

      scanLargeVectorRecursively_CL(blockSums[level], blockSums[level], blockSums, totalNumBlocks, level+1, type, init, deviceID);

      clSetKernelArg(scanUpdate, 0, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(scanUpdate, 1, sizeof(cl_mem), (void*)&blockSums_p);
      clSetKernelArg(scanUpdate, 2, sizeof(int), (void*)&isInclusive);
      clSetKernelArg(scanUpdate, 3, sizeof(T), (void*)&tempInit);
      clSetKernelArg(scanUpdate, 4, sizeof(size_t), (void*)&numElements);
      clSetKernelArg(scanUpdate, 5, sizeof(cl_mem), (void*)&ret_p);
      clSetKernelArg(scanUpdate, 6, updateSharedMemSize, NULL);

      err = clEnqueueNDRangeKernel(m_scanUpdateKernels_CL.at(deviceID).second->getQueue(), scanUpdate, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching Scan kernel!! " <<err <<"\n");
      }
   }
   else
   {
      clSetKernelArg(scanKernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(scanKernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(scanKernel, 2, sizeof(cl_mem), (void*)&blockSums_p);
      clSetKernelArg(scanKernel, 3, sizeof(size_t), (void*)&sharedElementsPerBlock);
      clSetKernelArg(scanKernel, 4, sizeof(size_t), (void*)&numElements);
      clSetKernelArg(scanKernel, 5, sharedMemSize, NULL);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      err = clEnqueueNDRangeKernel(m_scanKernels_CL.at(deviceID).second->getQueue(), scanKernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching Scan kernel!! " <<err <<"\n");
      }

      clSetKernelArg(scanUpdate, 0, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(scanUpdate, 1, sizeof(cl_mem), (void*)&blockSums_p);
      clSetKernelArg(scanUpdate, 2, sizeof(int), (void*)&isInclusive);
      clSetKernelArg(scanUpdate, 3, sizeof(T), (void*)&tempInit);
      clSetKernelArg(scanUpdate, 4, sizeof(size_t), (void*)&numElements);
      clSetKernelArg(scanUpdate, 5, sizeof(cl_mem), (void*)&ret_p);
      clSetKernelArg(scanUpdate, 6, updateSharedMemSize, NULL);

      err = clEnqueueNDRangeKernel(m_scanUpdateKernels_CL.at(deviceID).second->getQueue(), scanUpdate, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching Scan kernel!! " <<err <<"\n");
      }
   }

#if SKEPU_NUMGPU != 1
   ret_mem_p.changeDeviceData();
   ret_mem_p.copyDeviceToHost();
#endif

   return ret;
}

/*!
 *  Performs the Scan on an input range using \em OpenCL with a separate output range. Only one device is used for the scan.
 *  Allocates space for intermediate results from each block, and then calls scanLargeVectorRecursively_CL.
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
void Scan<ScanFunc>::scanSingle_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, unsigned int deviceID)
{
   // Setup parameters
   size_t numElements = inputEnd-inputBegin;
   size_t numBlocks;
   size_t numThreads = m_execPlan->maxThreads(numElements);
   const size_t numElementsPerThread = 1;

   std::vector<DeviceMemPointer_CL<typename InputIterator::value_type>*> blockSums;

   size_t numEl = numElements;

   do
   {
      numBlocks = numEl/(numThreads*numElementsPerThread) + (numEl%(numThreads*numElementsPerThread) == 0 ? 0:1);
      if (numBlocks >= 1)
      {
         blockSums.push_back(new DeviceMemPointer_CL<typename InputIterator::value_type>(NULL, numBlocks, m_environment->m_devices_CL.at(deviceID)));
      }
      numEl = numBlocks;
   }
   while (numEl > 1);

   typename InputIterator::device_pointer_type_cl in_mem_p = inputBegin.getParent().updateDevice_CL(inputBegin.getAddress(), numElements, m_environment->m_devices_CL.at(deviceID), true);
   typename OutputIterator::device_pointer_type_cl out_mem_p = outputBegin.getParent().updateDevice_CL(outputBegin.getAddress(), numElements, m_environment->m_devices_CL.at(deviceID), false);

   scanLargeVectorRecursively_CL(in_mem_p, out_mem_p, blockSums, numElements, 0, type, init, deviceID);

   out_mem_p->changeDeviceData();

   //Clean up
   for(size_t i = 0; i < blockSums.size(); ++i)
   {
      delete blockSums[i];
   }
}

/*!
 *  Performs the Scan on an input range using \em OpenCL with a separate output range. One or more devices can be
 *  used in the scan. The range is divided evenly among the participating devices which scans their part producing partial device results.
 *  The device results are scanned on the CPU before they are applied to each devices part.
 *  Allocates space for intermediate results from each block, and then calls scanLargeVectorRecursively_CL for each device.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param numDevices Integer deciding how many devices to utilize.
 */
template <typename ScanFunc>
template <typename InputIterator, typename OutputIterator>
void Scan<ScanFunc>::scanNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, size_t numDevices)
{
   cl_int err;

   size_t globalWorkSize[1];
   size_t localWorkSize[1];

   // Divide the elements amongst the devices
   size_t totalNumElements = inputEnd - inputBegin;
   size_t numElemPerSlice = totalNumElements / numDevices;
   size_t rest = totalNumElements % numDevices;
   typename InputIterator::value_type ret = 0;

   Vector<typename InputIterator::value_type> deviceSums;

   typename InputIterator::device_pointer_type_cl in_mem_p[MAX_GPU_DEVICES];
   typename OutputIterator::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   std::vector<DeviceMemPointer_CL<typename InputIterator::value_type>*> blockSums[MAX_GPU_DEVICES];

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElements;

      // If there is a rest, last device takse care of it
      if(i == numDevices-1)
         numElements = numElemPerSlice+rest;
      else
         numElements = numElemPerSlice;

      size_t numBlocks;
      size_t numThreads = m_execPlan->maxThreads(inputEnd-inputBegin);
      const size_t numElementsPerThread = 1;

      size_t numEl = numElements;

      do
      {
         numBlocks = numEl/(numThreads*numElementsPerThread) + (numEl%(numThreads*numElementsPerThread) == 0 ? 0:1);
         if (numBlocks >= 1)
         {
            blockSums[i].push_back(new DeviceMemPointer_CL<typename InputIterator::value_type>(NULL, numBlocks, m_environment->m_devices_CL.at(i)));
         }
         numEl = numBlocks;
      }
      while (numEl > 1);

      in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numElements, m_scanKernels_CL.at(i).second, false);
      out_mem_p[i] = outputBegin.getParent().updateDevice_CL((outputBegin+i*numElemPerSlice).getAddress(), numElements, m_scanKernels_CL.at(i).second, false);
   }

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElements;
      if(i == numDevices-1)
         numElements = numElemPerSlice+rest;
      else
         numElements = numElemPerSlice;

      size_t numBlocks;
      size_t numThreads = m_execPlan->maxThreads(inputEnd-inputBegin);
      const size_t numElementsPerThread = 1;

      size_t numEl = numElements;

      in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numElements, m_scanKernels_CL.at(i).second, true);

      ret = scanLargeVectorRecursively_CL(in_mem_p[i], out_mem_p[i], blockSums[i], numElements, 0, type, init, i);

      deviceSums.push_back(ret);

      out_mem_p[i]->changeDeviceData();
   }

   CPU(deviceSums, INCLUSIVE);

   // Add device sums to each devices data
   for(size_t i = 1; i < numDevices; ++i)
   {
      //Clean up
      for(size_t j = 0; j < blockSums[i].size(); ++j)
      {
         delete blockSums[i][j];
      }

      size_t numElements;
      if(i == numDevices-1)
         numElements = numElemPerSlice+rest;
      else
         numElements = numElemPerSlice;

      skepu::BackEndParams bp=m_execPlan->find_(inputEnd-inputBegin);
      size_t numThreads = bp.maxThreads;
      size_t maxBlocks = bp.maxBlocks;

      size_t numBlocks = std::min(numElements/(numThreads) + (numElements%(numThreads) == 0 ? 0:1), maxBlocks);

      typename OutputIterator::device_pointer_type_cl out_mem_p = outputBegin.getParent().updateDevice_CL((outputBegin+i*numElemPerSlice).getAddress(), numElements, m_scanKernels_CL.at(i).second, false);
      cl_mem out_p = out_mem_p->getDeviceDataPointer();

      cl_kernel scanAdd = m_scanAddKernels_CL.at(i).first;

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      clSetKernelArg(scanAdd, 0, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(scanAdd, 1, sizeof(typename InputIterator::value_type), (void*)&deviceSums[i-1]);
      clSetKernelArg(scanAdd, 2, sizeof(size_t), (void*)&numElements);

      err = clEnqueueNDRangeKernel(m_scanAddKernels_CL.at(i).second->getQueue(), scanAdd, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching scanAdd kernel!! " <<err <<"\n");
      }
   }
}

/*!
 *  Performs the Scan on a whole Vector using \em OpenCL with the input as output. A wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, int useNumGPU).
 *
 *  \param input A vector which will be scanned. It will be overwritten with the result.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::CL(Vector<T>& input, ScanType type, T init, int useNumGPU)
{
   CL(input.begin(), input.end(), input.begin(), type, init, useNumGPU);
}

/*!
 *  Performs the Scan on an input range using \em OpenCL with the input range also used as an output. A wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, int useNumGPU).
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename ScanFunc>
template <typename InputIterator>
void Scan<ScanFunc>::CL(InputIterator inputBegin, InputIterator inputEnd, ScanType type, typename InputIterator::value_type init, int useNumGPU)
{
   CL(inputBegin, inputEnd, inputBegin, type, init, useNumGPU);
}

/*!
 *  Performs the Scan on a whole Vector using \em OpenCL with a separate Vector as output. The output Vector will
 *  be resized an overwritten. A wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, int useNumGPU).
 *
 *  \param input A vector which will be scanned.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::CL(Vector<T>& input, Vector<T>& output, ScanType type, T init, int useNumGPU)
{
   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.size());
   }

   CL(input.begin(), input.end(), output.begin(), type, init, useNumGPU);
}

/*!
 *  Performs the Scan on a range of elements. Returns a scalar result. The function decides whether to perform
 *  the reduction on one device, calling scanSingle_CL or
 *  on multiple devices, calling scanNumDevices_CL.
 *  Using \em OpenCL as backend.
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
void Scan<ScanFunc>::CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("SCAN OPENCL\n")

   size_t numDevices = m_scanKernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      scanSingle_CL(inputBegin, inputEnd, outputBegin, type, init, 0);
   }
   else
   {
      scanNumDevices_CL(inputBegin, inputEnd, outputBegin, type, init, numDevices);
   }
}

}

#endif

