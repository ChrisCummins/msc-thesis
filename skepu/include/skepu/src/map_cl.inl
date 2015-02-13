/*! \file map_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the Map skeleton.
 */

#ifdef SKEPU_OPENCL

#include <iostream>
#include <typeinfo>
#include <string>

#include "operator_type.h"
#include "debug.h"
#include "map_kernels.h"

namespace skepu
{


/*!
 *  A function called by the constructor. It creates the OpenCL program for the skeleton and saves a handle for
 *  the kernel. The program is built from a string containing the user function (specified when constructing the
 *  skeleton) and a generic Map kernel. The type and function names in the generic kernel are relpaced by user function
 *  specific code before it is compiled by the OpenCL JIT compiler.
 *
 *  Also handles the use of doubles automatically by including "#pragma OPENCL EXTENSION cl_khr_fp64: enable" if doubles
 *  are used.
 */
template <typename MapFunc>
void Map<MapFunc>::createOpenCLProgram()
{
   //Creates the sourcecode
   std::string totalSource;
   std::string kernelSource;
   std::string funcSource = m_mapFunc->func_CL;
   std::string kernelName;

   if(m_mapFunc->datatype_CL == "double" || m_mapFunc->constype_CL == "double")
   {
      totalSource.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");
   }

   if(m_mapFunc->funcType == UNARY)
   {
      kernelSource = UnaryMapKernel_CL;
      kernelName = "UnaryMapKernel_" + m_mapFunc->funcName_CL;
   }
   else if(m_mapFunc->funcType == BINARY)
   {
      kernelSource = BinaryMapKernel_CL;
      kernelName = "BinaryMapKernel_" + m_mapFunc->funcName_CL;
   }
   else if(m_mapFunc->funcType == TERNARY)
   {
      kernelSource = TrinaryMapKernel_CL;
      kernelName = "TrinaryMapKernel_" + m_mapFunc->funcName_CL;
   }
   else
   {
      SKEPU_ERROR("Not valid function, should be Unary, Binary of Trinary!\n");
   }

   replaceTextInString(kernelSource, std::string("CONST_TYPE"), m_mapFunc->constype_CL);
   replaceTextInString(kernelSource, std::string("TYPE"), m_mapFunc->datatype_CL);
   replaceTextInString(kernelSource, std::string("KERNELNAME"), m_mapFunc->funcName_CL);
   replaceTextInString(kernelSource, std::string("FUNCTIONNAME"), m_mapFunc->funcName_CL);
   
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

      m_kernels_CL.push_back(std::make_pair(temp_kernel, (*it)));
   }
}

/*!
 *  Applies the Map skeleton to \em one range of elements specified by iterators. Result is saved to a seperate output range.
 *  The calculations can be performed by one or more devices. In the case of several, the input range is divided evenly
 *  amongst the participating devices.
 *
 *  The skeleton must have been created with a \em unary user function. \em OpenCL is used as backend.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element in the range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param numDevices Integer specifying the number of devices to perform the calculation on.
 */
template <typename MapFunc>
template <typename InputIterator, typename OutputIterator>
void Map<MapFunc>::mapNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, size_t numDevices)
{
   cl_int err;

   // Divide the elements amongst the devices
   size_t totalNumElements = inputEnd - inputBegin;
   size_t numElemPerSlice;
   size_t rest;

   numElemPerSlice = totalNumElements / numDevices;
   rest = totalNumElements % numDevices;

   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   typename InputIterator::device_pointer_type_cl in_mem_p[MAX_GPU_DEVICES];
   typename OutputIterator::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;

      // If there is a rest, last device takse care of it
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);
      out_mem_p[i] = outputBegin.getParent().updateDevice_CL((outputBegin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);
   }

   for(size_t i = 0; i < numDevices; ++i)
   {
      // If there is a rest, last device takse care of it
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      // Setup parameters
      BackEndParams bp;
      bp=m_execPlan->find_(inputEnd-inputBegin);
      size_t maxBlocks = bp.maxBlocks;
      size_t maxThreads = bp.maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      numThreads = std::min(maxThreads, (size_t)numElem);
      numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

      // Copies the elements to the device
      in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, true);

      cl_mem in_p = in_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 3, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}


/*!
 *  Performs mapping on \em one vector with \em  OpenCL backend. Input is used as output. The Map skeleton needs to
 *  be created with a \em unary user function. The function is a wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(Vector<T>& input, int useNumGPU)
{
   CL(input.begin(), input.end(), input.begin(), useNumGPU);
}

/*!
 *  Performs the Map on \em one vector with \em  OpenCL as backend. Seperate output vector is used. The Map skeleton needs to
 *  be created with a \em unary user function. The function is a wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 *  \param output The result vector, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(Vector<T>& input, Vector<T>& output, int useNumGPU)
{
   CL(input.begin(), input.end(), output.begin(), useNumGPU);
}


/*!
 *  Performs mapping on \em one matrix with \em  OpenCL backend. Input is used as output. The Map skeleton needs to
 *  be created with a \em unary user function. The function is a wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input A matrix which the mapping will be performed on. It will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(Matrix<T>& input, int useNumGPU)
{
   CL(input.begin(), input.end(), input.begin(), useNumGPU);
}

/*!
 *  Performs the Map on \em one matrix with \em  OpenCL as backend. Seperate output matrix is used. The Map skeleton needs to
 *  be created with a \em unary user function. The function is a wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input A matrix which the mapping will be performed on. It will be overwritten with the result.
 *  \param output The result matrix, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(Matrix<T>& input, Matrix<T>& output, int useNumGPU)
{
   CL(input.begin(), input.end(), output.begin(), useNumGPU);
}

/*!
 *  Performs the Map on \em one element range with \em  OpenCL as backend. Input is used as output. The Map skeleton needs to
 *  be created with a \em unary user function. The function is a wrapper for
 *  CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename InputIterator>
void Map<MapFunc>::CL(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU)
{
   CL(inputBegin, inputEnd, inputBegin, useNumGPU);
}

/*!
 *  Performs the Map on \em one element range with \em  OpenCL as backend. Seperate output range. The Map skeleton needs to
 *  be created with a \em unary user function. Calls mapNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int numDevices).
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename InputIterator, typename OutputIterator>
void Map<MapFunc>::CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP OPENCL\n")

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   mapNumDevices_CL(inputBegin, inputEnd, outputBegin, numDevices); // as here same method is invoked no matter how many GPUs we use
}

/*!
 *  Applies the Map skeleton to \em two ranges of elements specified by iterators. Result is saved to a seperate output range.
 *  The calculations can be performed by one or more devices. In the case of several, the input range is divided evenly
 *  amongst the participating devices.
 *
 *  The skeleton must have been created with a \em binary user function. \em  OpenCL is used as backend.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param numDevices Integer specifying the number of devices to perform the calculation on.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
void Map<MapFunc>::mapNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, size_t numDevices)
{
   cl_int err;

   // Divide the elements amongst the devices
   size_t totalNumElements = input1End - input1Begin;
   size_t numElemPerSlice = totalNumElements / numDevices;
   size_t rest = totalNumElements % numDevices;

   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   typename Input1Iterator::device_pointer_type_cl in1_mem_p[MAX_GPU_DEVICES];
   typename Input2Iterator::device_pointer_type_cl in2_mem_p[MAX_GPU_DEVICES];
   typename OutputIterator::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;

      // If there is a rest, last device takse care of it
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      in1_mem_p[i] = input1Begin.getParent().updateDevice_CL((input1Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);
      in2_mem_p[i] = input2Begin.getParent().updateDevice_CL((input2Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);
      out_mem_p[i] = outputBegin.getParent().updateDevice_CL((outputBegin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);
   }

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;

      // If there is a rest, last device takse care of it
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      // Setup parameters
      BackEndParams bp=m_execPlan->find_(numElem);
      size_t maxBlocks = bp.maxBlocks;
      size_t maxThreads = bp.maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      numThreads = std::min(maxThreads, (size_t)numElem);
      numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

      // Copies the elements to the device
      in1_mem_p[i] = input1Begin.getParent().updateDevice_CL((input1Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, true);
      in2_mem_p[i] = input2Begin.getParent().updateDevice_CL((input2Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, true);

      cl_mem in1_p = in1_mem_p[i]->getDeviceDataPointer();
      cl_mem in2_p = in2_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 4, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}

/*!
 *  Performs the Map on \em two vectors with \em  OpenCL as backend. Seperate output vector. The Map skeleton needs to
 *  be created with a \em binary user function. The function is a wrapper for
 *  CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(Vector<T>& input1, Vector<T>& input2, Vector<T>& output, int useNumGPU)
{
   CL(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin(), useNumGPU);
}


/*!
 *  Performs the Map on \em two matrices with \em  OpenCL as backend. Seperate output matrix. The Map skeleton needs to
 *  be created with a \em binary user function. The function is a wrapper for
 *  CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& output, int useNumGPU)
{
   CL(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin(), useNumGPU);
}



/*!
 *  Performs the Map on \em two element ranges with \em  OpenCL as backend. Seperate output range. The Map skeleton needs to
 *  be created with a \em binary user function. Calls
 *  mapNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int numDevices).
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
void Map<MapFunc>::CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP OPENCL\n")

   size_t n = input1End - input1Begin;

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   mapNumDevices_CL(input1Begin, input1End, input2Begin, input2End, outputBegin, numDevices); // as here same method is invoked no matter how many GPUs we use
}

/*!
 *  Applies the Map skeleton to \em three ranges of elements specified by iterators. Result is saved to a seperate output range.
 *  The calculations can be performed by one or more devices. In the case of several, the input range is divided evenly
 *  amongst the participating devices.
 *
 *  The skeleton must have been created with a \em trinary user function. \em OpenCL is used as backend.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param input3Begin An iterator to the first element in the third range.
 *  \param input3End An iterator to the last element of the third range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param numDevices Integer specifying the number of devices to perform the calculation on.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
void Map<MapFunc>::mapNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, size_t numDevices)
{
   cl_int err;

   // Divide the elements amongst the devices
   size_t totalNumElements = input1End - input1Begin;
   size_t numElemPerSlice = totalNumElements / numDevices;
   size_t rest = totalNumElements % numDevices;

   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   typename Input1Iterator::device_pointer_type_cl in1_mem_p[MAX_GPU_DEVICES];
   typename Input2Iterator::device_pointer_type_cl in2_mem_p[MAX_GPU_DEVICES];
   typename Input3Iterator::device_pointer_type_cl in3_mem_p[MAX_GPU_DEVICES];
   typename OutputIterator::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;

      // If there is a rest, last device takse care of it
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      in1_mem_p[i] = input1Begin.getParent().updateDevice_CL((input1Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);
      in2_mem_p[i] = input2Begin.getParent().updateDevice_CL((input2Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);
      in3_mem_p[i] = input3Begin.getParent().updateDevice_CL((input3Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);
      out_mem_p[i] = outputBegin.getParent().updateDevice_CL((outputBegin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);
   }

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;

      // If there is a rest, last device takse care of it
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      // Setup parameters
      BackEndParams bp=m_execPlan->find_(input1End-input1Begin);
      size_t maxBlocks = bp.maxBlocks;
      size_t maxThreads = bp.maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      numThreads = std::min(maxThreads, (size_t)numElem);
      numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

      // Copies the elements to the device
      in1_mem_p[i] = input1Begin.getParent().updateDevice_CL( (input1Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, true);
      in2_mem_p[i] = input2Begin.getParent().updateDevice_CL( (input2Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, true);
      in3_mem_p[i] = input3Begin.getParent().updateDevice_CL( (input3Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, true);

      cl_mem in1_p = in1_mem_p[i]->getDeviceDataPointer();
      cl_mem in2_p = in2_mem_p[i]->getDeviceDataPointer();
      cl_mem in3_p = in3_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&in3_p);
      clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 5, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}

/*!
 *  Performs the Map on \em three vectors with \em  OpenCL as backend. Seperate output vector. The Map skeleton needs to
 *  be created with a \em trinary user function. The function is a wrapper for
 *  CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param input3 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, Vector<T>& output, int useNumGPU)
{
   CL(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), output.begin(), useNumGPU);
}


/*!
 *  Performs the Map on \em three matrices with \em  OpenCL as backend. Seperate output matrix. The Map skeleton needs to
 *  be created with a \em trinary user function. The function is a wrapper for
 *  CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param input3 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, Matrix<T>& output, int useNumGPU)
{
   CL(input1.begin(), input1.end(), input2.begin(), input2.end(), input3.begin(), input3.end(), output.begin(), useNumGPU);
}

/*!
 *  Performs the Map on \em three element ranges with \em  OpenCL as backend. Seperate output range. The Map skeleton needs to
 *  be created with a \em trinary user function. Calls
 *  mapNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, int useNumGPU).
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
void Map<MapFunc>::CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP OPENCL\n")

   size_t n = input1End - input1Begin;

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   mapNumDevices_CL(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, outputBegin, numDevices); // as here same method is invoked no matter how many GPUs we use
}

}

#endif

