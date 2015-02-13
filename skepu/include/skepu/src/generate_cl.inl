/*! \file generate_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the Generate skeleton.
 */

#ifdef SKEPU_OPENCL

#include <iostream>

#include "operator_type.h"

#include "generate_kernels.h"

namespace skepu
{


/*!
 *  A function called by the constructor. It creates the OpenCL program for the skeleton and saves a handle for
 *  the kernel. The program is built from a string containing the user function (specified when constructing the
 *  skeleton) and a generic Generate kernel. The type and function names in the generic kernel are relpaced by user function
 *  specific code before it is compiled by the OpenCL JIT compiler.
 *
 *  Also handles the use of doubles automatically by including "#pragma OPENCL EXTENSION cl_khr_fp64: enable" if doubles
 *  are used.
 */
template <typename GenerateFunc>
void Generate<GenerateFunc>::createOpenCLProgram()
{
   //Creates the sourcecode
   std::string totalSource;
   std::string kernelSource;
   std::string funcSource = m_generateFunc->func_CL;
   std::string kernelName;

   if(m_generateFunc->datatype_CL == "double" || m_generateFunc->constype_CL == "double")
   {
      totalSource.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");
   }

   if(m_generateFunc->funcType == GENERATE)
   {
      kernelSource = GenerateKernel_CL;
      kernelName = "GenerateKernel_" + m_generateFunc->funcName_CL;
   }
   else if(m_generateFunc->funcType == GENERATE_MATRIX)
   {
      kernelSource = GenerateKernel_CL_Matrix;
      kernelName = "GenerateKernel_Matrix_" + m_generateFunc->funcName_CL;
   }
   else
   {
      SKEPU_ERROR("Not valid function, should be a Generate function!\n");
   }

   replaceTextInString(kernelSource, std::string("CONST_TYPE"), m_generateFunc->constype_CL);
   replaceTextInString(kernelSource, std::string("TYPE"), m_generateFunc->datatype_CL);
   replaceTextInString(kernelSource, std::string("KERNELNAME"), m_generateFunc->funcName_CL);
   replaceTextInString(kernelSource, std::string("FUNCTIONNAME"), m_generateFunc->funcName_CL);
   
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
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output range. For the \em OpenCL backend.
 *  The calculations can be performed by one or more devices. In the case of several, the input range is divided evenly
 *  amongst the participating devices.
 *
 *  \param numElements The number of elements to be generated.
 *  \param outputBegin An iterator pointing to the first element in the range which will be overwritten with generated values.
 *  \param numDevices Integer specifying the number of devices to perform the calculation on.
 */
template <typename GenerateFunc>
template <typename OutputIterator>
void Generate<GenerateFunc>::generateNumDevices_CL(size_t numElements, OutputIterator outputBegin, size_t numDevices)
{
   cl_int err;

   // Divide the elements amongst the devices
   size_t totalNumElements = numElements;
   size_t numElemPerSlice = totalNumElements / numDevices;
   size_t rest = totalNumElements % numDevices;
   typename GenerateFunc::CONST_TYPE const1 = m_generateFunc->getConstant();

   typename OutputIterator::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;

      // If there is a rest, last device takse care of it
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      out_mem_p[i] = outputBegin.getParent().updateDevice_CL((outputBegin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);
   }

   for(size_t i = 0; i < numDevices; ++i)
   {
      // If there is a rest, last device takes care of it
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      // Setup parameters
      skepu::BackEndParams bp=m_execPlan->find_(numElements);
      size_t maxBlocks = bp.maxBlocks;
      size_t maxThreads = bp.maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      numThreads = std::min(maxThreads, (size_t)numElem);
      numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      size_t indexOffset = i*numElemPerSlice;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 1, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&indexOffset);
      clSetKernelArg(kernel, 3, sizeof(typename OutputIterator::value_type), (void*)&const1);


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
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output vector which is resized to numElements. A wrapper for
 *  CL(size_t numElements, OutputIterator outputBegin, int useNumGPU). For the \em OpenCL backend.
 *
 *  \param numElements The number of elements to be generated.
 *  \param output The output vector which will be overwritten with the generated values.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename GenerateFunc>
template <typename T>
void Generate<GenerateFunc>::CL(size_t numElements, Vector<T>& output, int useNumGPU)
{
   if(output.size() != numElements)
   {
      output.clear();
      output.resize(numElements);
   }

   CL(numElements, output.begin(), useNumGPU);
}




/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output matrix which is resized to numElements. For the \em OpenCL backend.
 *
 *  \param numRows The number of rows to be generated.
 *  \param numCols The number of columns to be generated.
 *  \param output The output matrix which will be overwritten with the generated values.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename GenerateFunc>
template <typename T>
void Generate<GenerateFunc>::CL(size_t numRows, size_t numCols, Matrix<T>& output, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("GENERATE OPENCL Matrix\n")

   if((output.total_rows() != numRows) && (output.total_cols() != numCols))
   {
      output.clear();
      output.resize(numRows, numCols);
   }

   typename GenerateFunc::CONST_TYPE const1 = m_generateFunc->getConstant();
   cl_int err;

   size_t numDevices = m_kernels_CL.size();
   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }

   size_t numRowsPerSlice = numRows / numDevices;
   size_t restRows = numRows % numDevices;

   size_t numElemPerSlice = numRowsPerSlice*numCols;
   size_t restElem = restRows * numCols;

   typename Matrix<T>::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   // First create OpenCL memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+restElem;
      else
         numElem = numElemPerSlice;

      out_mem_p[i] = output.updateDevice_CL((output.getAddress()+i*numElemPerSlice), numElem, m_kernels_CL.at(i).second, false);
   }

   size_t yoffset[MAX_GPU_DEVICES];
   size_t xsize[MAX_GPU_DEVICES];
   size_t ysize[MAX_GPU_DEVICES];

   // Fill out argument struct with right information and start threads.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+restElem;
      else
         numElem = numElemPerSlice;

      size_t nrows = numElem/numCols;

      // Setup parameters
      size_t globalWorkSize[2];
      size_t localWorkSize[2];
      localWorkSize[0] =  (numCols>32)? 32: numCols;                            // each thread does multiple Xs
      localWorkSize[1] =  (nrows>16)? 16: nrows;
      globalWorkSize[0] = ((numCols+(localWorkSize[0]-1)) / localWorkSize[0]) * localWorkSize[0];
      globalWorkSize[1] = ((nrows+(localWorkSize[1]-1))  / localWorkSize[1]) * localWorkSize[1];

      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      yoffset[i] = (numElemPerSlice/numCols)*i;
      xsize[i] = numCols;
      ysize[i] = nrows;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 1, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&xsize[i]);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&ysize[i]);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&yoffset[i]);
      clSetKernelArg(kernel, 5, sizeof(typename GenerateFunc::CONST_TYPE), (void*)&const1);

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!! " <<err <<"\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}

/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output range. For the \em OpenCL backend. Calls
 *  generateNumDevices_CL(size_t numElements, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param numElements The number of elements to be generated.
 *  \param outputBegin An iterator pointing to the first element in the range which will be overwritten with generated values.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename GenerateFunc>
template <typename OutputIterator>
void Generate<GenerateFunc>::CL(size_t numElements, OutputIterator outputBegin, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("GENERATE OPENCL\n")

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   generateNumDevices_CL(numElements, outputBegin, numDevices); // as here same method is invoked no matter how many GPUs we use
}


}

#endif

