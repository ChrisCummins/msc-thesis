/*! \file reduce_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the Reduce skeleton.
 */

#ifdef SKEPU_OPENCL

#include <iostream>

#include "reduce_kernels.h"
#include "operator_type.h"

#include "device_mem_pointer_cl.h"

namespace skepu
{


/*!
 *  A function called by the constructor. It creates the OpenCL program for the skeleton and saves a handle for
 *  the kernel. The program is built from a string containing the user function (specified when constructing the
 *  skeleton) and a generic Reduce kernel. The type and function names in the generic kernel are relpaced by user function
 *  specific code before it is compiled by the OpenCL JIT compiler.
 *
 *  Also handles the use of doubles automatically by including "#pragma OPENCL EXTENSION cl_khr_fp64: enable" if doubles
 *  are used.
 */
template <typename ReduceFunc>
void Reduce<ReduceFunc, ReduceFunc>::createOpenCLProgram()
{
   //Creates the sourcecode
   std::string totalSource;
   std::string kernelSource;
   std::string funcSource = m_reduceFunc->func_CL;
   std::string kernelName;

   if(m_reduceFunc->datatype_CL == "double")
   {
      totalSource.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");
   }

   kernelSource = ReduceKernel_CL;
   kernelName = "ReduceKernel_" + m_reduceFunc->funcName_CL;

   replaceTextInString(kernelSource, std::string("TYPE"), m_reduceFunc->datatype_CL);
   replaceTextInString(kernelSource, std::string("KERNELNAME"), m_reduceFunc->funcName_CL);
   replaceTextInString(kernelSource, std::string("FUNCTIONNAME"), m_reduceFunc->funcName_CL);

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
         SKEPU_ERROR("Error creating a kernel!! name: " <<kernelName.c_str() <<"\n");
      }

      m_kernels_CL.push_back(std::make_pair(temp_kernel, (*it)));
   }
}




/*!
 *  Performs the Reduction on non-zero elements of a SparseMatrix with \em OpenCL as backend. Returns a scalar result. The function
 *  uses only \em one device which is decided by a parameter. A Helper method.
 *
 *  \param input A sparse matrix on which the reduction will be performed on.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::reduceSingle_CL(SparseMatrix<T> &input, unsigned int deviceID)
{
   cl_int err;

   // Setup parameters
   size_t n = input.total_nnz();

   size_t maxThreads = 256;
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   size_t globalWorkSize[1];
   size_t localWorkSize[1];

   T result = 0;

   getNumBlocksAndThreads(n, maxBlocks, maxThreads, numBlocks, numThreads);

   // Decide size of shared memory
   size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);

   // Copies the elements to the device
   typename SparseMatrix<T>::device_pointer_type_cl in_mem_p = input.updateDevice_CL(input.get_values(), n, m_kernels_CL.at(deviceID).second, true);

   // Create the output memory
   DeviceMemPointer_CL<T> out_mem_p(&result, numBlocks, m_kernels_CL.at(deviceID).second);

   cl_kernel kernel = m_kernels_CL.at(deviceID).first;
   cl_mem in_p = in_mem_p->getDeviceDataPointer();
   cl_mem out_p = out_mem_p.getDeviceDataPointer();

   // Sets the kernel arguments for first reduction
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 3, sharedMemSize, NULL);

   globalWorkSize[0] = numBlocks * numThreads;
   localWorkSize[0] = numThreads;

   // First reduce all elements blockwise so that each block produces one element.
   err = clEnqueueNDRangeKernel(m_kernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching kernel!!\n");
   }

   // Sets the kernel arguments for second reduction
   n = numBlocks;
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 3, sharedMemSize, NULL);

   globalWorkSize[0] = 1 * numThreads;
   localWorkSize[0] = numThreads;

   // Reduces the elements from the previous reduction in a single block to produce the scalar result.
   err = clEnqueueNDRangeKernel(m_kernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
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
 *  Performs the Reduction on a whole Vector. Returns a scalar result. A wrapper for CL(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU).
 *  Using \em OpenCL as backend.
 *
 *  \param input A vector which the reduction will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::CL(Vector<T>& input, int useNumGPU)
{
   return CL(input.begin(), input.end(), useNumGPU);
}

/*!
 *  Performs the Reduction on a whole Matrix. Returns a scalar result. A wrapper for CL(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU).
 *  Using \em OpenCL as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::CL(Matrix<T>& input, int useNumGPU)
{
   return CL(input.begin(), input.end(), useNumGPU);
}



/*!
 *  Performs the Reduction on non-zero elements of a SparseMatrix. Returns a scalar result.
 *  Using \em OpenCL as backend.
 *
 *  \param input A sparse matrix which the reduction will be performed on.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::CL(SparseMatrix<T>& input, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE SparseMatrix OPENCL\n")

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      return reduceSingle_CL(input, 0);
   }
   else
   {
      cl_int err;

      // Divide elements among participating devices
      size_t totalNumElements = input.total_nnz();
      size_t numElemPerSlice = totalNumElements / numDevices;
      size_t rest = totalNumElements % numDevices;

      T result[MAX_GPU_DEVICES];

      typename SparseMatrix<T>::device_pointer_type_cl in_mem_p[MAX_GPU_DEVICES]; // as vector, matrix and sparse matrix have common types
      typename SparseMatrix<T>::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

      T *values = input.get_values();

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

         in_mem_p[i] = input.updateDevice_CL((values+i*numElemPerSlice), numElem, m_kernels_CL.at(i).second, false);

         // Create the output memory
         out_mem_p[i] = new DeviceMemPointer_CL<T>(&result[i], numBlocks[i], m_kernels_CL.at(i).second);
      }

      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         // Decide size of shared memory
         size_t sharedMemSize = (numThreads[i] <= 32) ? 2 * numThreads[i] * sizeof(T) : numThreads[i] * sizeof(T);

         // Copies the elements to the device
         in_mem_p[i] = input.updateDevice_CL((values+i*numElemPerSlice), numElem, m_kernels_CL.at(i).second, true);

         cl_kernel kernel = m_kernels_CL.at(i).first;
         cl_mem in_p = in_mem_p[i]->getDeviceDataPointer();
         cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

         // Sets the kernel arguments for first reduction
         clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
         clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
         clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&numElem);
         clSetKernelArg(kernel, 3, sharedMemSize, NULL);

         globalWorkSize[0] = numBlocks[i] * numThreads[i];
         localWorkSize[0] = numThreads[i];

         // First reduce all elements blockwise so that each block produces one element.
         err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
         if(err != CL_SUCCESS)
         {
            SKEPU_ERROR("Error launching kernel!!\n");
         }

         // Sets the kernel arguments for second reduction
         numElem = numBlocks[i];
         clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
         clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
         clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&numElem);
         clSetKernelArg(kernel, 3, sharedMemSize, NULL);

         globalWorkSize[0] = 1 * numThreads[i];
         localWorkSize[0] = numThreads[i];

         // Reduces the elements from the previous reduction in a single block to produce the scalar result.
         err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
         if(err != CL_SUCCESS)
         {
            SKEPU_ERROR("Error launching kernel!!\n");
         }

         //Copy back result
         out_mem_p[i]->changeDeviceData();
      }

      // Reduces results from each device on the CPU to yield the total result.
      out_mem_p[0]->copyDeviceToHost(1);
      T totalResult = result[0];
      for(size_t i = 1; i < numDevices; ++i)
      {
         out_mem_p[i]->copyDeviceToHost(1);
         totalResult = m_reduceFunc->CPU(totalResult, result[i]);

         // Clean up
         delete out_mem_p[i];
      }

      return totalResult;
   }
}
















/*!
 *  Performs the Reduction, either row-wise or column-wise, on a Matrix with \em OpenCL as backend.
 *  Has an output parameter \em SkePU vector of reduction result. The function
 *  uses only \em one device which is decided by a parameter. A Helper method.
 *
 *  \param input A matrix on which the reduction will be performed on.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
void Reduce<ReduceFunc, ReduceFunc>::reduceSingleThreadOneDim_CL(Matrix<T> &input, unsigned int deviceID, Vector<T> &result)
{
   cl_int err;

   size_t rows = input.total_rows();
   size_t cols = input.total_cols();
   size_t size = rows * cols;

   // Setup parameters
   size_t maxThreads = 256;
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   Device_CL *device = m_kernels_CL.at(deviceID).second;

   getNumBlocksAndThreads(cols, maxBlocks, maxThreads, numBlocks, numThreads);

   // Decide size of shared memory
   size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);

   // Copies the elements to the device all at once, better(?)
   typename Matrix<T>::device_pointer_type_cl in_mem_p = input.updateDevice_CL(input.getAddress(), size, device, true);

   // Manually allocate output memory in this case, if only 1 block allocate for two
   cl_mem deviceMemPointer = allocateOpenCLMemory<T>(rows*numBlocks, m_kernels_CL.at(deviceID).second);

   cl_mem deviceInPointer = in_mem_p->getDeviceDataPointer();

   cl_mem d_input = deviceInPointer;
   cl_mem d_output = deviceMemPointer;

   cl_kernel kernel = m_kernels_CL.at(deviceID).first;

   cl_buffer_region info;

   // First reduce all elements row-wise so that each row produces one element.
   for(size_t r=0; r<rows; r++)
   {
      if(r>0)
      {
         // (origin, size) defines the offset and size in bytes in buffer.
         info.origin = r*cols*sizeof(T);
         info.size = cols*sizeof(T);
         d_input = clCreateSubBuffer(deviceInPointer, (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);
         info.origin = r*numBlocks*sizeof(T);
         info.size = numBlocks*sizeof(T);
         d_output = clCreateSubBuffer(deviceMemPointer, (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);
      }
      // execute the reduction for the given row
      ExecuteReduceOnADevice<T>(cols, numThreads, numBlocks, d_input, d_output, kernel, device);

      // Now get the reduction result back for the given row
      copyDeviceToHost<T>(&result[r], d_output, 1, device, 0);

      if(r>0)
      {
         // Should delete the buffers allocated....
         freeOpenCLMemory<T>(d_input);
         freeOpenCLMemory<T>(d_output);
      }
   }

   freeOpenCLMemory<T>(deviceMemPointer);
}






/*!
 *  Performs the Reduction, either row-wise or column-wise, on a Matrix. Returns a \em SkePU vector of reduction result.
 *  Using \em OpenCL as backend.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \param reducePolicy The policy specifying how reduction will be performed, can be either REDUCE_ROW_WISE_ONLY of REDUCE_COL_WISE_ONLY
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
Vector<T> Reduce<ReduceFunc, ReduceFunc>::CL(Matrix<T>& input, ReducePolicy reducePolicy, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE Matrix[] OPENCL\n")

   Matrix<T> *matrix = NULL;

   if(reducePolicy==REDUCE_COL_WISE_ONLY)
      matrix = &(~input);
   else // assume  reducePolict==REDUCE_ROW_WISE_ONLY)
      matrix = &input;

   size_t rows = matrix->total_rows();
   size_t cols = matrix->total_cols();

   skepu::Vector<T> result(rows);

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      reduceSingleThreadOneDim_CL(*matrix, 0, result);
      return result;
   }
   else
   {
      size_t maxThreads = 256;  // number of threads per block, taken from NVIDIA source
      size_t maxBlocks = 64;

      size_t numRowsPerSlice = rows / numDevices;
      size_t restRows = rows % numDevices;

      typename Matrix<T>::device_pointer_type_cl in_mem_p[MAX_GPU_DEVICES];

      cl_mem deviceMemPointers[MAX_GPU_DEVICES];

      // Setup parameters
      size_t numThreads = 0;
      size_t numBlocks = 0;

      getNumBlocksAndThreads(cols, maxBlocks, maxThreads, numBlocks, numThreads); // first do it for each column

      cl_buffer_region info;

      cl_kernel kernel;
      Device_CL *device;

      // First create OpenCL memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         device = m_kernels_CL.at(i).second;

         size_t numRows;
         if(i == numDevices-1)
            numRows = numRowsPerSlice+restRows;
         else
            numRows = numRowsPerSlice;

         in_mem_p[i] = matrix->updateDevice_CL((matrix->getAddress()+i*numRowsPerSlice*cols), numRows*cols, device, false);


         size_t outSize=numRows*numBlocks;

         // Manually allocate output memory in this case
         deviceMemPointers[i] = allocateOpenCLMemory<T>(outSize, device);
      }

      cl_mem d_input = NULL;
      cl_mem d_output = NULL;

      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numRows;
         if(i == numDevices-1)
            numRows = numRowsPerSlice+restRows;
         else
            numRows = numRowsPerSlice;

         kernel = m_kernels_CL.at(i).first;
         device = m_kernels_CL.at(i).second;

         in_mem_p[i] = matrix->updateDevice_CL((matrix->getAddress()+i*numRowsPerSlice*cols), numRows*cols, device, true);

         d_input = in_mem_p[i]->getDeviceDataPointer();
         d_output = deviceMemPointers[i];

         cl_mem deviceInPointer = d_input;

         // First reduce all elements row-wise so that each row produces one element.
         for(size_t r=0; r<numRows; r++)
         {
            if(r>0)
            {
               // (origin, size) defines the offset and size in bytes in buffer.
               info.origin = r*cols*sizeof(T);
               info.size = cols*sizeof(T);
               d_input = clCreateSubBuffer(deviceInPointer, (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);
               info.origin = r*numBlocks*sizeof(T);
               info.size = numBlocks*sizeof(T);
               d_output = clCreateSubBuffer(deviceMemPointers[i], (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);
            }
            // execute the reduction for the given row
            ExecuteReduceOnADevice<T>(cols, numThreads, numBlocks, d_input, d_output, kernel, device);

            // Now get the reduction result back for the given row
            copyDeviceToHost<T>(&result[r+(numRowsPerSlice*i)], d_output, 1, device, 0);

            if(r>0)
            {
               // Should delete the buffers allocated....
               freeOpenCLMemory<T>(d_input);
               freeOpenCLMemory<T>(d_output);
            }
         }
      }

      finishAll();

      for(size_t i = 0; i < numDevices; ++i)
      {
         freeOpenCLMemory<T>(deviceMemPointers[i]);
      }


      return result;
   }
}


























/*!
 *  Performs the Reduction, either row-wise or column-wise, on non-zero elements of a SparseMatrix with \em OpenCL as backend.
 *  The function uses only \em one device which is decided by a parameter. A Helper method.
 *
 *  \param input A sparse matrix on which the reduction will be performed on.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
void Reduce<ReduceFunc, ReduceFunc>::reduceSingleThreadOneDim_CL(SparseMatrix<T> &input, unsigned int deviceID, Vector<T> &result)
{
   cl_int err;

   T *resultPtr = result.getAddress();

   // Setup parameters
   size_t rows = input.total_rows();
   size_t size = input.total_nnz();

   size_t avgElemPerRow = size/rows;

   // Setup parameters
   size_t maxThreads = 256;
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   Device_CL *device = m_kernels_CL.at(deviceID).second;

   getNumBlocksAndThreads(avgElemPerRow, maxBlocks, maxThreads, numBlocks, numThreads);

   // Decide size of shared memory
   size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);

   // Copies the elements to the device all at once, better(?)
   typename SparseMatrix<T>::device_pointer_type_cl in_mem_p = input.updateDevice_CL(input.get_values(), size, device, true);

   std::vector<T> tempResult(rows);

   // Manually allocate output memory in this case,
   cl_mem deviceMemPointer = allocateOpenCLMemory<T>(rows*numBlocks, m_kernels_CL.at(deviceID).second);

   cl_mem deviceInPointer = in_mem_p->getDeviceDataPointer();

   cl_mem d_input = deviceInPointer;
   cl_mem d_output = deviceMemPointer;

   cl_kernel kernel = m_kernels_CL.at(deviceID).first;

   cl_buffer_region info;

   size_t elemPerRow = 0, offsetInfo = 0;

   // First reduce all elements row-wise so that each row produces one element.
   for(size_t r=0; r<rows; r++)
   {
      elemPerRow = input.get_rowSize(r);

      if(r>0 && elemPerRow>1)
      {
         // (origin, size) defines the offset and size in bytes in buffer.
         info.origin = offsetInfo*sizeof(T);
         info.size = elemPerRow*sizeof(T);
         d_input = clCreateSubBuffer(deviceInPointer, (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);
         info.origin = r*numBlocks*sizeof(T);
         info.size = numBlocks*sizeof(T);
         d_output = clCreateSubBuffer(deviceMemPointer, (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);
      }

      if(elemPerRow>1)
      {
         // execute the reduction for the given row
         ExecuteReduceOnADevice<T>(elemPerRow, numThreads, numBlocks, d_input, d_output, kernel, device);

         // Now get the reduction result back for the given row
         copyDeviceToHost<T>(&resultPtr[r], d_output, 1, device, 0);

         if(r>0)
         {
            freeOpenCLMemory<T>(d_input);
            freeOpenCLMemory<T>(d_output);
         }
      }
      else
         resultPtr[r] = ((elemPerRow>0) ? (input.begin(r)(0)):T()); // dont use [] operator as that internally invalidate device copy

      offsetInfo += elemPerRow;
   }

   freeOpenCLMemory<T>(deviceMemPointer);
}





/*!
 *  Performs the Reduction on a whole SparseMatrix. Returns a \em SkePU vector of reduction result.
 *  Using \em OpenCL as backend.
 *
 *  \param input A sparse matrix which the reduction will be performed on.
 *  \param reducePolicy The policy specifying how reduction will be performed, can be either REDUCE_ROW_WISE_ONLY of REDUCE_COL_WISE_ONLY
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
Vector<T> Reduce<ReduceFunc, ReduceFunc>::CL(SparseMatrix<T>& input, ReducePolicy reducePolicy, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE SparseMatrix[] OPENCL\n")

   SparseMatrix<T> *matrix = NULL;

   if(reducePolicy==REDUCE_COL_WISE_ONLY)
      matrix = &(~input);
   else // assume  reducePolict==REDUCE_ROW_WISE_ONLY)
      matrix = &input;

   size_t rows = matrix->total_rows();

   skepu::Vector<T> result(rows);

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      reduceSingleThreadOneDim_CL(*matrix, 0, result);
      return result;
   }
   else
   {
      T *resultPtr = result.getAddress();

      size_t size = matrix->total_nnz();

      size_t avgElemPerRow = size/rows;

      size_t maxThreads = 256;
      size_t maxBlocks = 64;

      size_t numRowsPerSlice = rows / numDevices;
      size_t restRows = rows % numDevices;

      typename SparseMatrix<T>::device_pointer_type_cl in_mem_p[MAX_GPU_DEVICES];

      cl_mem deviceMemPointers[MAX_GPU_DEVICES];

      // Setup parameters
      size_t numThreads = 0;
      size_t numBlocks = 0;

      getNumBlocksAndThreads(avgElemPerRow, maxBlocks, maxThreads, numBlocks, numThreads); // first do it for each column

      cl_buffer_region info;

      cl_kernel kernel;
      Device_CL *device;

      size_t offset = 0, end = 0;

      // First create OpenCL memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         device = m_kernels_CL.at(i).second;

         size_t numRows;
         if(i == numDevices-1)
            numRows = numRowsPerSlice+restRows;
         else
            numRows = numRowsPerSlice;

         end = matrix->get_rowOffsetFromStart(numRows+i*numRowsPerSlice);

         in_mem_p[i] = matrix->updateDevice_CL((matrix->get_values()+offset), end-offset, device, false);


         size_t outSize=numRows*numBlocks;

         // Manually allocate output memory in this case
         deviceMemPointers[i] = allocateOpenCLMemory<T>(outSize, device);

         offset = end;
      }

      std::vector<T> tempResult(rows);

      cl_mem d_input = NULL;
      cl_mem d_output = NULL;

      offset = end = 0;

      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numRows;
         if(i == numDevices-1)
            numRows = numRowsPerSlice+restRows;
         else
            numRows = numRowsPerSlice;

         kernel = m_kernels_CL.at(i).first;
         device = m_kernels_CL.at(i).second;

         end = matrix->get_rowOffsetFromStart(numRows+i*numRowsPerSlice);

         in_mem_p[i] = matrix->updateDevice_CL((matrix->get_values()+offset), end-offset, device, true);

         d_input = in_mem_p[i]->getDeviceDataPointer();
         d_output = deviceMemPointers[i];

         cl_mem deviceInPointer = d_input;

         size_t elemPerRow = 0;

         size_t offsetInfo = 0;

         // First reduce all elements row-wise so that each row produces one element.
         for(size_t r=0; r<numRows; r++)
         {
            elemPerRow = matrix->get_rowSize(r+(numRowsPerSlice*i));

            if(r>0 && elemPerRow > 1)
            {
               // (origin, size) defines the offset and size in bytes in buffer.
               info.origin = offsetInfo*sizeof(T);
               info.size = elemPerRow*sizeof(T);
               d_input = clCreateSubBuffer(deviceInPointer, (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);
               info.origin = r*numBlocks*sizeof(T);
               info.size = numBlocks*sizeof(T);
               d_output = clCreateSubBuffer(deviceMemPointers[i], (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);
            }
            if(elemPerRow > 1)
            {
               // execute the reduction for the given row
               ExecuteReduceOnADevice<T>(elemPerRow, numThreads, numBlocks, d_input, d_output, kernel, device);

               // Now get the reduction result back for the given row
               copyDeviceToHost<T>(&resultPtr[r+(numRowsPerSlice*i)], d_output, 1, device, 0);

               if(r>0)
               {
                  freeOpenCLMemory<T>(d_input);
                  freeOpenCLMemory<T>(d_output);
               }
            }
            else
               resultPtr[r+(numRowsPerSlice*i)] = ((elemPerRow>0) ? (matrix->begin(r+(numRowsPerSlice*i))(0)):T()); // dont use [] operator as that internally invalidate device copy

            offsetInfo += elemPerRow;
         }

         offset = end;
      }

      finishAll();

      // Free allocated memory on all devices
      for(size_t i = 0; i < numDevices; ++i)
      {
         freeOpenCLMemory<T>(deviceMemPointers[i]);
      }

      return result;
   }
}






















/*!
 *  Performs the Reduction on a range of elements with \em OpenCL as backend. Returns a scalar result. The function
 *  uses only \em one device which is decided by a parameter. A Helper method.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type Reduce<ReduceFunc, ReduceFunc>::reduceSingle_CL(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID)
{
   cl_int err;

   // Setup parameters
   size_t n = inputEnd-inputBegin;
   size_t maxThreads = 256; //m_execPlan->maxThreads(n);
   size_t maxBlocks = 64; //maxThreads;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   size_t globalWorkSize[1];
   size_t localWorkSize[1];

   typename InputIterator::value_type result = 0;

   getNumBlocksAndThreads(n, maxBlocks, maxThreads, numBlocks, numThreads);

   // Decide size of shared memory
   size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(typename InputIterator::value_type) : numThreads * sizeof(typename InputIterator::value_type);

   // Copies the elements to the device
   typename InputIterator::device_pointer_type_cl in_mem_p = inputBegin.getParent().updateDevice_CL(inputBegin.getAddress(), n, m_kernels_CL.at(deviceID).second, true);

   // Create the output memory
   DeviceMemPointer_CL<typename InputIterator::value_type> out_mem_p(&result, numBlocks, m_kernels_CL.at(deviceID).second);

   cl_kernel kernel = m_kernels_CL.at(deviceID).first;
   cl_mem in_p = in_mem_p->getDeviceDataPointer();
   cl_mem out_p = out_mem_p.getDeviceDataPointer();

   // Sets the kernel arguments for first reduction
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 3, sharedMemSize, NULL);

   globalWorkSize[0] = numBlocks * numThreads;
   localWorkSize[0] = numThreads;

   // First reduce all elements blockwise so that each block produces one element.
   err = clEnqueueNDRangeKernel(m_kernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching kernel!!\n");
   }

   // Sets the kernel arguments for second reduction
   n = numBlocks;
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 3, sharedMemSize, NULL);

   globalWorkSize[0] = 1 * numThreads;
   localWorkSize[0] = numThreads;

   // Reduces the elements from the previous reduction in a single block to produce the scalar result.
   err = clEnqueueNDRangeKernel(m_kernels_CL.at(deviceID).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
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
 *  Performs the Reduction on a range of elements. Returns a scalar result. The function decides whether to perform
 *  the reduction on one device, calling reduceSingle_CL(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID) or
 *  on multiple devices, calling reduceNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, size_t numDevices).
 *  Using \em OpenCL as backend.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type Reduce<ReduceFunc, ReduceFunc>::CL(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE OPENCL\n")

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      return reduceSingle_CL(inputBegin, inputEnd, 0);
   }
   else
   {
      cl_int err;

      size_t n = inputEnd - inputBegin;

      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename InputIterator::value_type result[MAX_GPU_DEVICES];

      typename InputIterator::device_pointer_type_cl in_mem_p[MAX_GPU_DEVICES];

      typename InputIterator::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

      // Setup parameters
      size_t maxThreads = 256;
      size_t maxBlocks = 64;

      size_t numThreads[MAX_GPU_DEVICES];
      size_t numBlocks[MAX_GPU_DEVICES];

      // First create OpenCL memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         numBlocks[i] = numThreads[i] = 0;

         getNumBlocksAndThreads(numElem, maxBlocks, maxThreads, numBlocks[i], numThreads[i]);

         in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);

         // Create the output memory
         out_mem_p[i] = new DeviceMemPointer_CL<typename InputIterator::value_type>(&result[i], numBlocks[i], m_kernels_CL.at(i).second);
      }

      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      // Create argument structs for all threads
      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         // Copies the elements to the device
         in_mem_p[i] = inputBegin.getParent().updateDevice_CL((inputBegin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, true);

         // Decide size of shared memory
         size_t sharedMemSize = (numThreads[i] <= 32) ? 2 * numThreads[i] * sizeof(typename InputIterator::value_type) : numThreads[i] * sizeof(typename InputIterator::value_type);

         cl_kernel kernel = m_kernels_CL.at(i).first;
         cl_mem in_p = in_mem_p[i]->getDeviceDataPointer();
         cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

         // Sets the kernel arguments for first reduction
         clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
         clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
         clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&numElem);
         clSetKernelArg(kernel, 3, sharedMemSize, NULL);

         globalWorkSize[0] = numBlocks[i] * numThreads[i];
         localWorkSize[0] = numThreads[i];

         // First reduce all elements blockwise so that each block produces one element.
         err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
         if(err != CL_SUCCESS)
         {
            SKEPU_ERROR("Error launching kernel!!\n");
         }

         // Sets the kernel arguments for second reduction
         numElem = numBlocks[i];
         clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
         clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
         clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&numElem);
         clSetKernelArg(kernel, 3, sharedMemSize, NULL);

         globalWorkSize[0] = 1 * numThreads[i];
         localWorkSize[0] = numThreads[i];

         // Reduces the elements from the previous reduction in a single block to produce the scalar result.
         err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
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

         //Clean up
         delete out_mem_p[i];
      }

      return totalResult;
   }
}

}

#endif

