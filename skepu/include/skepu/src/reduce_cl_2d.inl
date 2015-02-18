/*! \file reduce_cl_2d.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the 2DReduce skeleton.
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
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
void Reduce<ReduceFuncRowWise, ReduceFuncColWise>::createOpenCLProgram()
{
   //Creates the sourcecode
   std::string totalSource;
   std::string kernelSourceRowWise, kernelSourceColWise;
   std::string funcSource = m_reduceFuncRowWise->func_CL + m_reduceFuncColWise->func_CL;
   std::string kernelNameRowWise, kernelNameColWise;

   if(m_reduceFuncRowWise->datatype_CL == "double")
   {
      totalSource.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");
   }
   if( (m_reduceFuncRowWise->datatype_CL)!=(m_reduceFuncRowWise->datatype_CL) )
   {
      std::cerr<<"ERROR! For 2D Reduce, data type of both user functions must be same\n";
      SKEPU_EXIT();
   }


   kernelSourceRowWise = kernelSourceColWise = ReduceKernel_CL;
   kernelNameRowWise = "ReduceKernel_" + m_reduceFuncRowWise->funcName_CL;
   kernelNameColWise = "ReduceKernel_" + m_reduceFuncColWise->funcName_CL;

   replaceTextInString(kernelSourceRowWise, std::string("TYPE"), m_reduceFuncRowWise->datatype_CL);
   replaceTextInString(kernelSourceRowWise, std::string("KERNELNAME"), m_reduceFuncRowWise->funcName_CL);
   replaceTextInString(kernelSourceRowWise, std::string("FUNCTIONNAME"), m_reduceFuncRowWise->funcName_CL);

   replaceTextInString(kernelSourceColWise, std::string("TYPE"), m_reduceFuncColWise->datatype_CL);
   replaceTextInString(kernelSourceColWise, std::string("KERNELNAME"), m_reduceFuncColWise->funcName_CL);
   replaceTextInString(kernelSourceColWise, std::string("FUNCTIONNAME"), m_reduceFuncColWise->funcName_CL);


   // check for extra user-supplied opencl code for custome datatype
   totalSource.append( read_file_into_string(OPENCL_SOURCE_FILE_NAME) );

   totalSource.append(funcSource);
   totalSource.append(kernelSourceRowWise);
   totalSource.append(kernelSourceColWise);
   
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

      temp_kernel = clCreateKernel(temp_program, kernelNameRowWise.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<kernelNameRowWise.c_str() <<"\n" <<err <<"\n");
      }

      m_kernels_CL_RowWise.push_back(std::make_pair(temp_kernel, (*it)));

      temp_kernel = clCreateKernel(temp_program, kernelNameColWise.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<kernelNameColWise.c_str() <<"\n" <<err <<"\n");
      }

      m_kernels_CL_ColWise.push_back(std::make_pair(temp_kernel, (*it)));
   }
}




/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  matrix by using \em OpenCL backend. Returns a scalar result. The function
 *  uses only \em one OpenCL device which is decided by a parameter.
 *
 *  \param input An input matrix whose elements need to be reduced.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::reduceSingle_CL(Matrix<T> &input, unsigned int deviceID)
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

   Device_CL *device = m_kernels_CL_RowWise.at(deviceID).second;

   getNumBlocksAndThreads(cols, maxBlocks, maxThreads, numBlocks, numThreads);

   // Decide size of shared memory
   size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);

   // Copies the elements to the device all at once, better(?)
   typename Matrix<T>::device_pointer_type_cl in_mem_p = input.updateDevice_CL(input.getAddress(), size, device, true);

   std::vector<T> tempResult(rows);


   // Manually allocate output memory in this case, if only 1 block allocate for two
   cl_mem deviceMemPointer = allocateOpenCLMemory<T>(rows*((numBlocks>1)?numBlocks:2), m_kernels_CL_RowWise.at(deviceID).second);

   cl_mem deviceInPointer = in_mem_p->getDeviceDataPointer();

   cl_mem d_input = deviceInPointer;
   cl_mem d_output = deviceMemPointer;

   cl_kernel kernel = m_kernels_CL_RowWise.at(deviceID).first;

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
      copyDeviceToHost<T>(&tempResult[r], d_output, 1, device, 0);


   }

   T result;

   // Do column-wise reduction
   kernel = m_kernels_CL_ColWise.at(deviceID).first;

   // if sufficient work then do final (column-wise) reduction on GPU
   if(rows>REDUCE_GPU_THRESHOLD)
   {
      clFinish(device->getQueue());

      // reset to starting position and use it as an input
      d_input = deviceMemPointer;

      info.origin = rows*sizeof(T);
      info.size = rows*sizeof(T);
      d_output = clCreateSubBuffer(deviceMemPointer, (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);

      copyHostToDevice<T>(&tempResult[0], d_input, rows, device, 0);

      getNumBlocksAndThreads(rows, maxBlocks, maxThreads, numBlocks, numThreads); // get numThreads and numBlocks for final reduction

      // execute the reduction for the resulting row
      ExecuteReduceOnADevice<T>(rows, numThreads, numBlocks, d_input, d_output, kernel, device);

      clFinish(device->getQueue());

      copyDeviceToHost<T>(&result, d_output, 1, device, 0);
   }
   else // do final reduction step on CPU instead
   {
      result = tempResult[0];

      for(size_t r=1; r<rows; r++)
      {
         result = m_reduceFuncColWise->CPU(result, tempResult[r]);
      }
   }

   freeOpenCLMemory<T>(deviceMemPointer);

   return result;
}

/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  matrix by using \em OpenCL backend. Returns a scalar result. The function
 *  uses a variable number of devices, dividing the range of elemets equally
 *  among the participating devices each reducing its part. The results are
 *  then reduced themselves on the CPU.
 *
 *  \param input An input matrix whose elements need to be reduced.
 *  \param numDevices Integer deciding how many devices to utilize.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::reduceNumDevices_CL(Matrix<T> &input, size_t numDevices)
{
   cl_int err;

   size_t rows = input.total_rows();
   size_t cols = input.total_cols();
   size_t size = rows * cols;

   size_t maxThreads = 256; //m_execPlan->maxThreads(n);
   size_t maxBlocks = 64; //maxThreads;

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
      device = m_kernels_CL_RowWise.at(i).second;

      size_t numRows;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      in_mem_p[i] = input.updateDevice_CL((input.getAddress()+i*numRowsPerSlice*cols), numRows*cols, device, false);


      size_t outSize=numRows*numBlocks;
      if(i==0 && outSize<(2*rows)) // for first device as later we may re-use this storage to do final reduction on GPU 0
         outSize = 2*rows; // make it at least that much large

      // Manually allocate output memory in this case
      deviceMemPointers[i] = allocateOpenCLMemory<T>(outSize, device);
   }

   std::vector<T> tempResult(rows);

   cl_mem d_input = NULL;
   cl_mem d_output = NULL;

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numRows;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      kernel = m_kernels_CL_RowWise.at(i).first;
      device = m_kernels_CL_RowWise.at(i).second;

      in_mem_p[i] = input.updateDevice_CL((input.getAddress()+i*numRowsPerSlice*cols), numRows*cols, device, true);

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
         copyDeviceToHost<T>(&tempResult[r+(numRowsPerSlice*i)], d_output, 1, device, 0);
      }
   }

   finishAll();

   T result;


   // if sufficient work then do final (column-wise) reduction on GPU
   if(rows>REDUCE_GPU_THRESHOLD)
   {
      // Do column-wise reduction on GPU 0
      kernel = m_kernels_CL_ColWise.at(0).first;
      device = m_kernels_CL_ColWise.at(0).second;

      // reset to starting position and use it as an input
      d_input = deviceMemPointers[0];

      info.origin = rows*sizeof(T);
      info.size = rows*sizeof(T);
      d_output = clCreateSubBuffer(deviceMemPointers[0], (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);

      copyHostToDevice<T>(&tempResult[0], d_input, rows, device, 0);

      getNumBlocksAndThreads(rows, maxBlocks, maxThreads, numBlocks, numThreads); // get numThreads and numBlocks for final reduction

      // execute the reduction for the resulting row
      ExecuteReduceOnADevice<T>(rows, numThreads, numBlocks, d_input, d_output, kernel, device);

      clFinish(device->getQueue());

      copyDeviceToHost<T>(&result, d_output, 1, device, 0);
   }
   else // do final reduction step on CPU instead
   {
      result = tempResult[0];

      for(size_t r=1; r<rows; r++)
      {
         result = m_reduceFuncColWise->CPU(result, tempResult[r]);
      }
   }

   // Free allocated memory on all devices
   for(size_t i = 0; i < numDevices; ++i)
   {
      freeOpenCLMemory<T>(deviceMemPointers[i]);
   }

   return result;
}



/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  matrix by using \em OpenCL backend. Returns a scalar result. The function
 *  can be applied by any number of OpenCL devices, thus internally calling the
 *  \em reduceSingle_CL or \em reduceNumDevices_CL depending upon number of
 *  OpenCL devices specified/available.
 *
 *  \param input An input matrix whose elements need to be reduced.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::CL(Matrix<T>& input, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE 2D Matrix OPENCL\n")

   size_t numDevices = m_kernels_CL_RowWise.size();

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
      return reduceNumDevices_CL(input, numDevices);
   }
}







/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  sparse matrix by using \em OpenCL backend. Returns a scalar result. The function
 *  uses only \em one OpenCL device which is decided by a parameter.
 *
 *  \param input An input sparse matrix whose elements need to be reduced.
 *  \param deviceID Integer deciding which device to utilize.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::reduceSingle_CL(SparseMatrix<T> &input, unsigned int deviceID)
{
   cl_int err;

   // Setup parameters
   size_t rows = input.total_rows();
   size_t size = input.total_nnz();

   size_t avgElemPerRow = size/rows;

   // Setup parameters
   size_t maxThreads = 256;
   size_t maxBlocks = 64;

   size_t numBlocks = 0;
   size_t numThreads = 0;

   Device_CL *device = m_kernels_CL_RowWise.at(deviceID).second;

   getNumBlocksAndThreads(avgElemPerRow, maxBlocks, maxThreads, numBlocks, numThreads);

   // Decide size of shared memory
   size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);

   // Copies the elements to the device all at once, better(?)
   typename SparseMatrix<T>::device_pointer_type_cl in_mem_p = input.updateDevice_CL(input.get_values(), size, device, true);

   std::vector<T> tempResult(rows);

   // Manually allocate output memory in this case, if only 1 block allocate for two
   cl_mem deviceMemPointer = allocateOpenCLMemory<T>(rows*((numBlocks>1)?numBlocks:2), m_kernels_CL_RowWise.at(deviceID).second);

   cl_mem deviceInPointer = in_mem_p->getDeviceDataPointer();

   cl_mem d_input = deviceInPointer;
   cl_mem d_output = deviceMemPointer;

   cl_kernel kernel = m_kernels_CL_RowWise.at(deviceID).first;

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
         copyDeviceToHost<T>(&tempResult[r], d_output, 1, device, 0);
      }
      else
         tempResult[r] = ((elemPerRow>0) ? (input.begin(r)(0)):T()); // dont use [] operator as that internally invalidate device copy

      offsetInfo += elemPerRow;
   }

   T result;

   // Do column-wise reduction
   kernel = m_kernels_CL_ColWise.at(deviceID).first;

   // if sufficient work then do final (column-wise) reduction on GPU
   if(rows>REDUCE_GPU_THRESHOLD)
   {
      clFinish(device->getQueue());

      // reset to starting position and use it as an input
      d_input = deviceMemPointer;

      info.origin = rows*sizeof(T);
      info.size = rows*sizeof(T);
      d_output = clCreateSubBuffer(deviceMemPointer, (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);

      copyHostToDevice<T>(&tempResult[0], d_input, rows, device, 0);

      getNumBlocksAndThreads(rows, maxBlocks, maxThreads, numBlocks, numThreads); // get numThreads and numBlocks for final reduction

      // execute the reduction for the resulting row
      ExecuteReduceOnADevice<T>(rows, numThreads, numBlocks, d_input, d_output, kernel, device);

      clFinish(device->getQueue());

      copyDeviceToHost<T>(&result, d_output, 1, device, 0);
   }
   else // do final reduction step on CPU instead
   {
      result = tempResult[0];

      for(size_t r=1; r<rows; r++)
      {
         result = m_reduceFuncColWise->CPU(result, tempResult[r]);
      }
   }

   freeOpenCLMemory<T>(deviceMemPointer);

   return result;
}

/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  sparse matrix by using \em OpenCL backend. Returns a scalar result. The function
 *  uses a variable number of devices, dividing the range of elemets equally
 *  among the participating devices each reducing its part. The results are
 *  then reduced themselves on the CPU.
 *
 *  \param input An input sparse matrix whose elements need to be reduced.
 *  \param numDevices Integer deciding how many devices to utilize.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::reduceNumDevices_CL(SparseMatrix<T> &input, size_t numDevices)
{
   cl_int err;

   size_t rows = input.total_rows();
   size_t size = input.total_nnz();

   size_t avgElemPerRow = size/rows;

   size_t maxThreads = 256; //m_execPlan->maxThreads(n);
   size_t maxBlocks = 64; //maxThreads;

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
      device = m_kernels_CL_RowWise.at(i).second;

      size_t numRows;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      end = input.get_rowOffsetFromStart(numRows+i*numRowsPerSlice);

      in_mem_p[i] = input.updateDevice_CL((input.get_values()+offset), end-offset, device, false);


      size_t outSize=numRows*numBlocks;
      if(i==0 && outSize<(2*rows)) // for first device as later we may re-use this storage to do final reduction on GPU 0
         outSize = 2*rows; // make it at least that much large

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

      kernel = m_kernels_CL_RowWise.at(i).first;
      device = m_kernels_CL_RowWise.at(i).second;

      end = input.get_rowOffsetFromStart(numRows+i*numRowsPerSlice);

      in_mem_p[i] = input.updateDevice_CL((input.get_values()+offset), end-offset, device, true);

      d_input = in_mem_p[i]->getDeviceDataPointer();
      d_output = deviceMemPointers[i];

      cl_mem deviceInPointer = d_input;

      size_t elemPerRow = 0;

      size_t offsetInfo = 0;

      // First reduce all elements row-wise so that each row produces one element.
      for(size_t r=0; r<numRows; r++)
      {
         elemPerRow = input.get_rowSize(r+(numRowsPerSlice*i));

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
            copyDeviceToHost<T>(&tempResult[r+(numRowsPerSlice*i)], d_output, 1, device, 0);
         }
         else
            tempResult[r+(numRowsPerSlice*i)] = ((elemPerRow>0) ? (input.begin(r+(numRowsPerSlice*i))(0)):T()); // dont use [] operator as that internally invalidate device copy

         offsetInfo += elemPerRow;
      }

      offset = end;
   }

   finishAll();

   T result;


   // if sufficient work then do final (column-wise) reduction on GPU
   if(rows>REDUCE_GPU_THRESHOLD)
   {
      // Do column-wise reduction on GPU 0
      kernel = m_kernels_CL_ColWise.at(0).first;
      device = m_kernels_CL_ColWise.at(0).second;

      // reset to starting position and use it as an input
      d_input = deviceMemPointers[0];

      info.origin = rows*sizeof(T);
      info.size = rows*sizeof(T);
      d_output = clCreateSubBuffer(deviceMemPointers[0], (int)CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &info, NULL);

      copyHostToDevice<T>(&tempResult[0], d_input, rows, device, 0);

      getNumBlocksAndThreads(rows, maxBlocks, maxThreads, numBlocks, numThreads); // get numThreads and numBlocks for final reduction

      // execute the reduction for the resulting row
      ExecuteReduceOnADevice<T>(rows, numThreads, numBlocks, d_input, d_output, kernel, device);

      clFinish(device->getQueue());

      copyDeviceToHost<T>(&result, d_output, 1, device, 0);
   }
   else // do final reduction step on CPU instead
   {
      result = tempResult[0];

      for(size_t r=1; r<rows; r++)
      {
         result = m_reduceFuncColWise->CPU(result, tempResult[r]);
      }
   }


   // Free allocated memory on all devices
   for(size_t i = 0; i < numDevices; ++i)
   {
      freeOpenCLMemory<T>(deviceMemPointers[i]);
   }

   return result;
}



/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on an input
 *  sparse matrix by using \em OpenCL backend. Returns a scalar result. The function
 *  can be applied by any number of OpenCL devices, thus internally calling the
 *  \em reduceSingle_CL or \em reduceNumDevices_CL depending upon number of
 *  OpenCL devices specified/available.
 *
 *  \param input An input sparse matrix whose elements need to be reduced.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::CL(SparseMatrix<T>& input, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("REDUCE 2D SparseMatrix OPENCL\n")

   size_t numDevices = m_kernels_CL_RowWise.size();

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
      return reduceNumDevices_CL(input, numDevices);
   }
}







}

#endif

