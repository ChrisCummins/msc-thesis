/*! \file mapreduce_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the MapReduce skeleton.
 */

#ifdef SKEPU_OPENCL

#include <iostream>

#include "operator_type.h"

#include "mapreduce_kernels.h"
#include "reduce_kernels.h"
#include "device_mem_pointer_cl.h"

namespace skepu
{


/*!
 *  A function called by the constructor. It creates the OpenCL program for the skeleton and saves a handle for
 *  the kernel. The program is built from a string containing the user functions (specified when constructing the
 *  skeleton) and a generic MapReduce kernel. The type and function names in the generic kernel are relpaced by user function
 *  specific code before it is compiled by the OpenCL JIT compiler. Both a MapReduce kernel and a regular Reduce kernel are built.
 *
 *  Also handles the use of doubles automatically by including "#pragma OPENCL EXTENSION cl_khr_fp64: enable" if doubles
 *  are used.
 */
template <typename MapFunc, typename ReduceFunc>
void MapReduce<MapFunc, ReduceFunc>::createOpenCLProgram()
{
   //Creates the sourcecode
   std::string totalSource;

   std::string kernelSource;
   std::string funcSource;
   std::string kernelName;

   std::string reducekernelSource;
   std::string reducekernelName;

   //If same function, only use one
   if(m_reduceFunc->funcName_CL == m_mapFunc->funcName_CL)
   {
      funcSource = m_mapFunc->func_CL;
   }
   else
   {
      funcSource = m_mapFunc->func_CL + m_reduceFunc->func_CL;
   }

   if(m_reduceFunc->datatype_CL != m_mapFunc->datatype_CL)
   {
      SKEPU_ERROR("MapFunc and ReduceFunc must be of same type in MapReduce!\n");
   }

   if(m_reduceFunc->datatype_CL == "double")
   {
      totalSource.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");
   }

   if(m_mapFunc->funcType == UNARY)
   {
      kernelSource = UnaryMapReduceKernel_CL;
      kernelName = "UnaryMapReduceKernel_" + m_mapFunc->funcName_CL + m_reduceFunc->funcName_CL;
   }
   else if(m_mapFunc->funcType == BINARY)
   {
      kernelSource = BinaryMapReduceKernel_CL;
      kernelName = "BinaryMapReduceKernel_" + m_mapFunc->funcName_CL + m_reduceFunc->funcName_CL;
   }
   else if(m_mapFunc->funcType == TERNARY)
   {
      kernelSource = TrinaryMapReduceKernel_CL;
      kernelName = "TrinaryMapReduceKernel_" + m_mapFunc->funcName_CL + m_reduceFunc->funcName_CL;
   }
   else
   {
      SKEPU_ERROR("Not valid function, map function should be Unary, Binary of Trinary!\n");
   }

   if(m_reduceFunc->funcType == BINARY)
   {
      reducekernelSource = ReduceKernel_CL;
      reducekernelName = "ReduceKernel_" + m_reduceFunc->funcName_CL;
   }
   else
   {
      SKEPU_ERROR("Not valid function, reduce function should be Binary!\n");
   }

   replaceTextInString(kernelSource, std::string("CONST_TYPE"), m_mapFunc->constype_CL);
   replaceTextInString(kernelSource, std::string("TYPE"), m_mapFunc->datatype_CL);
   replaceTextInString(kernelSource, std::string("KERNELNAME"), m_mapFunc->funcName_CL + m_reduceFunc->funcName_CL);
   replaceTextInString(kernelSource, std::string("FUNCTIONNAME_MAP"), m_mapFunc->funcName_CL);
   replaceTextInString(kernelSource, std::string("FUNCTIONNAME_REDUCE"), m_reduceFunc->funcName_CL);
   
   replaceTextInString(reducekernelSource, std::string("TYPE"), m_reduceFunc->datatype_CL);
   replaceTextInString(reducekernelSource, std::string("KERNELNAME"), m_reduceFunc->funcName_CL);
   replaceTextInString(reducekernelSource, std::string("FUNCTIONNAME"), m_reduceFunc->funcName_CL);
   
   // check for extra user-supplied opencl code for custome datatype
   totalSource.append( read_file_into_string(OPENCL_SOURCE_FILE_NAME) );

   totalSource.append(funcSource);
   totalSource.append(kernelSource);
   totalSource.append(reducekernelSource);

   if(sizeof(size_t) <= 4)
      replaceTextInString(totalSource, std::string("size_t "), "unsigned int ");
   else if(sizeof(size_t) <= 8)
      replaceTextInString(totalSource, std::string("size_t "), "unsigned long ");
   else
      SKEPU_ERROR("OpenCL code compilation issue: sizeof(size_t) is bigger than 8 bytes: " << sizeof(size_t));

   DEBUG_TEXT_LEVEL3(totalSource);

   // Builds the code and creates kernel for all devices
   // Creates both a MapReduce and a Reduce kernel
   for(std::vector<Device_CL*>::iterator it = m_environment->m_devices_CL.begin(); it != m_environment->m_devices_CL.end(); ++it)
   {
      const char* c_src = totalSource.c_str();
      cl_int err;
      cl_program temp_program;
      cl_kernel temp_kernel;

      temp_program = clCreateProgramWithSource((*it)->getContext(), 1, &c_src, NULL, &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating OpenCL source!!\n");
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

      m_mapReduceKernels_CL.push_back(std::make_pair(temp_kernel, (*it)));

      temp_kernel = clCreateKernel(temp_program, reducekernelName.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<reducekernelName.c_str() <<"\n" <<err <<"\n");
      }

      m_reduceKernels_CL.push_back(std::make_pair(temp_kernel, (*it)));
   }
}

/*!
 *  Performs the Map on \em one range of elements and Reduce on the result with \em OpenCL as backend. Returns a scalar result.
 *  The function uses only \em one device which is decided by a parameter.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type MapReduce<MapFunc, ReduceFunc>::mapReduceSingle_CL(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID)
{
   cl_int err;

   // Setup parameters
   size_t n = inputEnd-inputBegin;
   size_t maxThreads = m_execPlan->maxThreads(n);
   size_t maxBlocks = maxThreads;
   size_t numBlocks;
   size_t numThreads;
   size_t globalWorkSize[1];
   size_t localWorkSize[1];
   typename InputIterator::value_type result = 0;
   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   numThreads = maxThreads;
   numBlocks = std::max((size_t) 1, std::min( (n / numThreads), maxBlocks));

   // Decide size of shared memory
   size_t sharedMemSize = sizeof(typename InputIterator::value_type) * numThreads;

   // Copies the elements to the device
   typename InputIterator::device_pointer_type_cl in_mem_p = inputBegin.getParent().updateDevice_CL(inputBegin.getAddress(), n, m_mapReduceKernels_CL.at(deviceID).second, true);

   // Create the output memory
   DeviceMemPointer_CL<typename InputIterator::value_type> out_mem_p(&result, maxBlocks, m_mapReduceKernels_CL.at(deviceID).second);

   cl_mem in_p = in_mem_p->getDeviceDataPointer();
   cl_mem out_p = out_mem_p.getDeviceDataPointer();

   cl_kernel kernel = m_mapReduceKernels_CL.at(deviceID).first;

   // Sets the kernel arguments for first map and reduce
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 3, sharedMemSize, NULL);
   clSetKernelArg(kernel, 4, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

   globalWorkSize[0] = numBlocks * numThreads;
   localWorkSize[0] = numThreads;

   // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
   err = clEnqueueNDRangeKernel(m_mapReduceKernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching kernel!!\n");
   }

   n = numBlocks;
   kernel = m_reduceKernels_CL.at(deviceID).first;

   // Sets the kernel arguments for second reduction
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 3, sharedMemSize, NULL);

   globalWorkSize[0] = 1 * numThreads;
   localWorkSize[0] = numThreads;

   // Reduces the elements from the previous reduction in a single block to produce the scalar result.
   // Here only a regular reduce kernel is needed since the mapping has been completed in previous step.
   err = clEnqueueNDRangeKernel(m_reduceKernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching kernel!!\n");
   }

   //Copy back result
   out_mem_p.changeDeviceData();
   out_mem_p.copyDeviceToHost(1);

   return result;
}

/*!
 *  Performs the Map on \em one range of elements and Reduce on the result with \em OpenCL as backend. Returns a scalar result. The function
 *  uses a variable number of devices, dividing the range of elemets equally among the participating devices each reducing
 *  its part. The results are then reduced themselves on the CPU.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param numDevices Integer deciding how many devices to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type MapReduce<MapFunc, ReduceFunc>::mapReduceNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, size_t numDevices)
{
   cl_int err;

   // Divide elements among participating devices
   size_t totalNumElements = inputEnd - inputBegin;
   size_t numElemPerSlice = totalNumElements / numDevices;
   size_t rest = totalNumElements % numDevices;
   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   typename InputIterator::value_type* result = new typename InputIterator::value_type[numDevices];
   DeviceMemPointer_CL<typename InputIterator::value_type>** out_mem_p = new DeviceMemPointer_CL<typename InputIterator::value_type>*[numDevices];

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      // Setup parameters
      size_t maxThreads = m_execPlan->maxThreads(inputEnd-inputBegin);
      size_t maxBlocks = maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      numThreads = maxThreads;
      numBlocks = std::max((size_t) 1, std::min( (numElem / numThreads), maxBlocks));

      // Decide size of shared memory
      size_t sharedMemSize = sizeof(typename InputIterator::value_type) * numThreads;

      // Copies the elements to the device
      typename InputIterator::device_pointer_type_cl in_mem_p = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numElem, m_mapReduceKernels_CL.at(i).second, true);

      // Create the output memory
      out_mem_p[i] = new DeviceMemPointer_CL<typename InputIterator::value_type>(&result[i], maxBlocks, m_mapReduceKernels_CL.at(i).second);

      cl_mem in_p = in_mem_p->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_mapReduceKernels_CL.at(i).first;

      // Sets the kernel arguments for first map and reduce
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 3, sharedMemSize, NULL);
      clSetKernelArg(kernel, 4, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
      err = clEnqueueNDRangeKernel(m_mapReduceKernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      numElem = numBlocks;
      kernel = m_reduceKernels_CL.at(i).first;

      // Sets the kernel arguments for second reduction
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 3, sharedMemSize, NULL);

      globalWorkSize[0] = 1 * numThreads;
      localWorkSize[0] = numThreads;

      // Reduces the elements from the previous reduction in a single block to produce the scalar result.
      // Here only a regular reduce kernel is needed since the mapping has been completed in previous step.
      err = clEnqueueNDRangeKernel(m_reduceKernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      //Copy back result
      out_mem_p[i]->changeDeviceData();
   }

   // Reduces results from each device on the CPU to yield the total result.
   out_mem_p[0]->copyDeviceToHost(1);
   typename InputIterator::value_type totalResult = result[0];
   for(size_t i = 1; i < numDevices; ++i)
   {
      out_mem_p[i]->copyDeviceToHost(1);
      totalResult = m_reduceFunc->CPU(totalResult, result[i]);
   }

   //Clean up
   for(size_t i = 0; i < numDevices; ++i)
   {
      delete out_mem_p[i];
   }
   delete[] out_mem_p;
   delete[] result;

   return totalResult;
}

/*!
 *  Performs the Map on \em one Vector and Reduce on the result. Returns a scalar result.
 *  A wrapper for CL(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU).
 *  Using \em OpenCL as backend.
 *
 *  \param input A vector which the map and reduce will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CL(Vector<T>& input, int useNumGPU)
{
   return CL(input.begin(), input.end(), useNumGPU);
}


/*!
 *  Performs the Map on \em one Matrix and Reduce on the result. Returns a scalar result.
 *  A wrapper for CL(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU).
 *  Using \em OpenCL as backend.
 *
 *  \param input A matrix which the map and reduce will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CL(Matrix<T>& input, int useNumGPU)
{
   return CL(input.begin(), input.end(), useNumGPU);
}

/*!
 *  Performs the Map on \em one range of elements and Reduce on the result. Returns a scalar result. The function decides whether to perform
 *  the map and reduce on one device, calling mapReduceSingle_CL(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID) or
 *  on multiple devices, calling mapReduceNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, size_t numDevices).
 *  Using \em OpenCL as backend.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type MapReduce<MapFunc, ReduceFunc>::CL(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE OPENCL\n")

   size_t numDevices = m_mapReduceKernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      return mapReduceSingle_CL(inputBegin, inputEnd, 0);
   }
   else
   {
      return mapReduceNumDevices_CL(inputBegin, inputEnd, numDevices);
   }
}

/*!
 *  Performs the Map on \em two ranges of elements and Reduce on the result with \em OpenCL as backend. Returns a scalar result.
 *  The function uses only \em one device which is decided by a parameter.
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
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::mapReduceSingle_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, unsigned int deviceID)
{
   cl_int err;

   // Setup parameters
   size_t n = input1End-input1Begin;
   size_t maxThreads = m_execPlan->maxThreads(n);
   size_t maxBlocks = maxThreads;
   size_t numBlocks;
   size_t numThreads;
   size_t globalWorkSize[1];
   size_t localWorkSize[1];
   typename Input1Iterator::value_type result = 0;
   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   numThreads = maxThreads;
   numBlocks = std::max((size_t) 1, std::min( (n / numThreads), maxBlocks));

   // Decide size of shared memory
   size_t sharedMemSize = sizeof(typename Input1Iterator::value_type) * numThreads;

   // Copies the elements to the device
   typename Input1Iterator::device_pointer_type_cl in1_mem_p = input1Begin.getParent().updateDevice_CL(input1Begin.getAddress(), n, m_mapReduceKernels_CL.at(deviceID).second, true);
   typename Input2Iterator::device_pointer_type_cl in2_mem_p = input2Begin.getParent().updateDevice_CL(input2Begin.getAddress(), n, m_mapReduceKernels_CL.at(deviceID).second, true);

   // Create the output memory
   DeviceMemPointer_CL<typename Input1Iterator::value_type> out_mem_p(&result, maxBlocks, m_mapReduceKernels_CL.at(deviceID).second);

   cl_mem in1_p = in1_mem_p->getDeviceDataPointer();
   cl_mem in2_p = in2_mem_p->getDeviceDataPointer();
   cl_mem out_p = out_mem_p.getDeviceDataPointer();

   cl_kernel kernel = m_mapReduceKernels_CL.at(deviceID).first;

   // Sets the kernel arguments for first map and reduce
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 4, sharedMemSize, NULL);
   clSetKernelArg(kernel, 5, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

   globalWorkSize[0] = numBlocks * numThreads;
   localWorkSize[0] = numThreads;

   // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
   err = clEnqueueNDRangeKernel(m_mapReduceKernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching kernel!!\n");
   }

   n = numBlocks;
   kernel = m_reduceKernels_CL.at(deviceID).first;

   // Sets the kernel arguments for second reduction
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 3, sharedMemSize, NULL);

   globalWorkSize[0] = 1 * numThreads;
   localWorkSize[0] = numThreads;

   // Reduces the elements from the previous reduction in a single block to produce the scalar result.
   // Here only a regular reduce kernel is needed since the mapping has been completed in previous step.
   err = clEnqueueNDRangeKernel(m_reduceKernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching kernel!!\n");
   }

   //Copy back result
   out_mem_p.changeDeviceData();
   out_mem_p.copyDeviceToHost(1);

   return result;
}

/*!
 *  Performs the Map on \em two ranges of elements and Reduce on the result with \em OpenCL as backend. Returns a scalar result. The function
 *  uses a variable number of devices, dividing the range of elemets equally among the participating devices each reducing
 *  its part. The results are then reduced themselves on the CPU.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param numDevices Integer deciding how many devices to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::mapReduceNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, size_t numDevices)
{
   cl_int err;

   // Divide elements among participating devices
   size_t totalNumElements = input1End - input1Begin;
   size_t numElemPerSlice = totalNumElements / numDevices;
   size_t rest = totalNumElements % numDevices;
   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   typename Input1Iterator::value_type* result = new typename Input1Iterator::value_type[numDevices];
   DeviceMemPointer_CL<typename Input1Iterator::value_type>** out_mem_p = new DeviceMemPointer_CL<typename Input1Iterator::value_type>*[numDevices];

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      // Setup parameters
      size_t maxThreads = m_execPlan->maxThreads(input1End-input1Begin);
      size_t maxBlocks = maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      numThreads = maxThreads;
      numBlocks = std::max((size_t) 1, std::min( (numElem / numThreads), maxBlocks));

      // Decide size of shared memory
      size_t sharedMemSize = sizeof(typename Input1Iterator::value_type) * numThreads;

      // Copies the elements to the device
      typename Input1Iterator::device_pointer_type_cl in1_mem_p = input1Begin.getParent().updateDevice_CL((input1Begin+i*numElemPerSlice).getAddress(), numElem, m_mapReduceKernels_CL.at(i).second, true);
      typename Input2Iterator::device_pointer_type_cl in2_mem_p = input2Begin.getParent().updateDevice_CL((input2Begin+i*numElemPerSlice).getAddress(), numElem, m_mapReduceKernels_CL.at(i).second, true);

      // Create the output memory
      out_mem_p[i] = new DeviceMemPointer_CL<typename Input1Iterator::value_type>(&result[i], maxBlocks, m_mapReduceKernels_CL.at(i).second);

      cl_mem in1_p = in1_mem_p->getDeviceDataPointer();
      cl_mem in2_p = in2_mem_p->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_mapReduceKernels_CL.at(i).first;

      // Sets the kernel arguments for first map and reduce
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 4, sharedMemSize, NULL);
      clSetKernelArg(kernel, 5, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
      err = clEnqueueNDRangeKernel(m_mapReduceKernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      numElem = numBlocks;
      kernel = m_reduceKernels_CL.at(i).first;

      // Sets the kernel arguments for second reduction
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 3, sharedMemSize, NULL);

      globalWorkSize[0] = 1 * numThreads;
      localWorkSize[0] = numThreads;

      // Reduces the elements from the previous reduction in a single block to produce the scalar result.
      // Here only a regular reduce kernel is needed since the mapping has been completed in previous step.
      err = clEnqueueNDRangeKernel(m_reduceKernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      //Copy back result
      out_mem_p[i]->changeDeviceData();
   }

   // Reduces results from each device on the CPU to yield the total result.
   out_mem_p[0]->copyDeviceToHost(1);
   typename Input1Iterator::value_type totalResult = result[0];
   for(size_t i = 1; i < numDevices; ++i)
   {
      out_mem_p[i]->copyDeviceToHost(1);
      totalResult = m_reduceFunc->CPU(totalResult, result[i]);
   }

   //Clean up
   for(size_t i = 0; i < numDevices; ++i)
   {
      delete out_mem_p[i];
   }
   delete[] out_mem_p;
   delete[] result;

   return totalResult;
}

/*!
 *  Performs the Map on \em two Vectors and Reduce on the result. Returns a scalar result.
 *  A wrapper for CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, int useNumGPU).
 *  Using \em OpenCL as backend.
 *
 *  \param input1 A Vector which the map and reduce will be performed on.
 *  \param input2 A Vector which the map and reduce will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CL(Vector<T>& input1, Vector<T>& input2, int useNumGPU)
{
   return CL(input1.begin(), input1.end(), input2.begin(), input2.end(), useNumGPU);
}


/*!
 *  Performs the Map on \em two matrices and Reduce on the result. Returns a scalar result.
 *  A wrapper for CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, int useNumGPU).
 *  Using \em OpenCL as backend.
 *
 *  \param input1 A matrix which the map and reduce will be performed on.
 *  \param input2 A matrix which the map and reduce will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CL(Matrix<T>& input1, Matrix<T>& input2, int useNumGPU)
{
   return CL(input1.begin(), input1.end(), input2.begin(), input2.end(), useNumGPU);
}

/*!
 *  Performs the Map on \em two ranges of elements and Reduce on the result. Returns a scalar result. The function decides whether to perform
 *  the map and reduce on one device, calling mapReduceSingle_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, unsigned int deviceID) or
 *  on multiple devices, calling mapReduceNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, size_t numDevices).
 *  Using \em OpenCL as backend.
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
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE OPENCL\n")

   size_t numDevices = m_mapReduceKernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      return mapReduceSingle_CL(input1Begin, input1End, input2Begin, input2End, 0);
   }
   else
   {
      return mapReduceNumDevices_CL(input1Begin, input1End, input2Begin, input2End, numDevices);
   }
}

/*!
 *  Performs the Map on \em three ranges of elements and Reduce on the result with \em OpenCL as backend. Returns a scalar result.
 *  The function uses only \em one device which is decided by a parameter.
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
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::mapReduceSingle_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, unsigned int deviceID)
{
   cl_int err;

   // Setup parameters
   size_t n = input1End-input1Begin;
   size_t maxThreads = m_execPlan->maxThreads(n);
   size_t maxBlocks = maxThreads;
   size_t numBlocks;
   size_t numThreads;
   size_t globalWorkSize[1];
   size_t localWorkSize[1];
   typename Input1Iterator::value_type result = 0;
   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   numThreads = maxThreads;
   numBlocks = std::max((size_t) 1, std::min( (n / numThreads), maxBlocks));

   // Decide size of shared memory
   size_t sharedMemSize = sizeof(typename Input1Iterator::value_type) * numThreads;

   // Copies the elements to the device
   typename Input1Iterator::device_pointer_type_cl in1_mem_p = input1Begin.getParent().updateDevice_CL(input1Begin.getAddress(), n, m_mapReduceKernels_CL.at(deviceID).second, true);
   typename Input2Iterator::device_pointer_type_cl in2_mem_p = input2Begin.getParent().updateDevice_CL(input2Begin.getAddress(), n, m_mapReduceKernels_CL.at(deviceID).second, true);
   typename Input3Iterator::device_pointer_type_cl in3_mem_p = input3Begin.getParent().updateDevice_CL(input3Begin.getAddress(), n, m_mapReduceKernels_CL.at(deviceID).second, true);

   // Create the output memory
   DeviceMemPointer_CL<typename Input1Iterator::value_type> out_mem_p(&result, maxBlocks, m_mapReduceKernels_CL.at(deviceID).second);

   cl_mem in1_p = in1_mem_p->getDeviceDataPointer();
   cl_mem in2_p = in2_mem_p->getDeviceDataPointer();
   cl_mem in3_p = in3_mem_p->getDeviceDataPointer();
   cl_mem out_p = out_mem_p.getDeviceDataPointer();

   cl_kernel kernel = m_mapReduceKernels_CL.at(deviceID).first;

   // Sets the kernel arguments for first map and reduce
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
   clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&in3_p);
   clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 5, sharedMemSize, NULL);
   clSetKernelArg(kernel, 6, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

   globalWorkSize[0] = numBlocks * numThreads;
   localWorkSize[0] = numThreads;

   // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
   err = clEnqueueNDRangeKernel(m_mapReduceKernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching kernel!!\n");
   }

   n = numBlocks;
   kernel = m_reduceKernels_CL.at(deviceID).first;

   // Sets the kernel arguments for second reduction
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 3, sharedMemSize, NULL);

   globalWorkSize[0] = 1 * numThreads;
   localWorkSize[0] = numThreads;

   // Reduces the elements from the previous reduction in a single block to produce the scalar result.
   // Here only a regular reduce kernel is needed since the mapping has been completed in previous step.
   err = clEnqueueNDRangeKernel(m_reduceKernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching kernel!!\n");
   }

   //Copy back result
   out_mem_p.changeDeviceData();
   out_mem_p.copyDeviceToHost(1);

   return result;
}

/*!
 *  Performs the Map on \em three ranges of elements and Reduce on the result with \em OpenCL as backend. Returns a scalar result. The function
 *  uses a variable number of devices, dividing the range of elemets equally among the participating devices each reducing
 *  its part. The results are then reduced themselves on the CPU.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param input3Begin An iterator to the first element in the third range.
 *  \param input3End An iterator to the last element of the third range.
 *  \param numDevices Integer deciding how many devices to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::mapReduceNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, size_t numDevices)
{
   cl_int err;

   // Divide elements among participating devices
   size_t totalNumElements = input1End - input1Begin;
   size_t numElemPerSlice = totalNumElements / numDevices;
   size_t rest = totalNumElements % numDevices;
   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   typename Input1Iterator::value_type* result = new typename Input1Iterator::value_type[numDevices];
   DeviceMemPointer_CL<typename Input1Iterator::value_type>** out_mem_p = new DeviceMemPointer_CL<typename Input1Iterator::value_type>*[numDevices];

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      // Setup parameters
      size_t maxThreads = m_execPlan->maxThreads(input1End-input1Begin);
      size_t maxBlocks = maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      numThreads = maxThreads;
      numBlocks = std::max((size_t) 1, std::min( (numElem / numThreads), maxBlocks));

      // Decide size of shared memory
      size_t sharedMemSize = sizeof(typename Input1Iterator::value_type) * numThreads;

      // Copies the elements to the device
      typename Input1Iterator::device_pointer_type_cl in1_mem_p = input1Begin.getParent().updateDevice_CL((input1Begin+i*numElemPerSlice).getAddress(), numElem, m_mapReduceKernels_CL.at(i).second, true);
      typename Input2Iterator::device_pointer_type_cl in2_mem_p = input2Begin.getParent().updateDevice_CL((input2Begin+i*numElemPerSlice).getAddress(), numElem, m_mapReduceKernels_CL.at(i).second, true);
      typename Input3Iterator::device_pointer_type_cl in3_mem_p = input3Begin.getParent().updateDevice_CL((input3Begin+i*numElemPerSlice).getAddress(), numElem, m_mapReduceKernels_CL.at(i).second, true);

      // Create the output memory
      out_mem_p[i] = new DeviceMemPointer_CL<typename Input1Iterator::value_type>(&result[i], maxBlocks, m_mapReduceKernels_CL.at(i).second);

      cl_mem in1_p = in1_mem_p->getDeviceDataPointer();
      cl_mem in2_p = in2_mem_p->getDeviceDataPointer();
      cl_mem in3_p = in3_mem_p->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_mapReduceKernels_CL.at(i).first;

      // Sets the kernel arguments for first map and reduce
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&in3_p);
      clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 5, sharedMemSize, NULL);
      clSetKernelArg(kernel, 6, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // First map and reduce all elements blockwise so that each block produces one element. After this the mapping is complete
      err = clEnqueueNDRangeKernel(m_mapReduceKernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      numElem = numBlocks;
      kernel =  m_reduceKernels_CL.at(i).first;

      // Sets the kernel arguments for second reduction
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 3, sharedMemSize, NULL);

      globalWorkSize[0] = 1 * numThreads;
      localWorkSize[0] = numThreads;

      // Reduces the elements from the previous reduction in a single block to produce the scalar result.
      // Here only a regular reduce kernel is needed since the mapping has been completed in previous step.
      err = clEnqueueNDRangeKernel(m_reduceKernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      //Copy back result
      out_mem_p[i]->changeDeviceData();
   }

   // Reduces results from each device on the CPU to yield the total result.
   out_mem_p[0]->copyDeviceToHost(1);
   typename Input1Iterator::value_type totalResult = result[0];
   for(size_t i = 1; i < numDevices; ++i)
   {
      out_mem_p[i]->copyDeviceToHost(1);
      totalResult = m_reduceFunc->CPU(totalResult, result[i]);
   }

   //Clean up
   for(size_t i = 0; i < numDevices; ++i)
   {
      delete out_mem_p[i];
   }
   delete[] out_mem_p;
   delete[] result;

   return totalResult;
}

/*!
 *  Performs the Map on \em three Vectors and Reduce on the result. Returns a scalar result.
 *  A wrapper for CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, int useNumGPU).
 *  Using \em OpenCL as backend.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param input3 A vector which the mapping will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CL(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, int useNumGPU)
{
   return CL(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), useNumGPU);
}


/*!
 *  Performs the Map on \em three matrices and Reduce on the result. Returns a scalar result.
 *  A wrapper for CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, int useNumGPU).
 *  Using \em OpenCL as backend.
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param input3 A matrix which the mapping will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::CL(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, int useNumGPU)
{
   return CL(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), useNumGPU);
}

/*!
 *  Performs the Map on \em three ranges of elements and Reduce on the result. Returns a scalar result. The function decides whether to perform
 *  the map and reduce on one device, calling mapReduceSingle_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, unsigned int deviceID) or
 *  on multiple devices, calling mapReduceNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, size_t numDevices).
 *  Using \em OpenCL as backend.
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
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPREDUCE OPENCL\n")

   size_t numDevices = m_mapReduceKernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      return mapReduceSingle_CL(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, 0);
   }
   else
   {
      return mapReduceNumDevices_CL(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, numDevices);
   }
}

}

#endif

