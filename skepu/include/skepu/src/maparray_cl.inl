/*! \file maparray_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the MapArray skeleton.
 */

#ifdef SKEPU_OPENCL

#include <iostream>

#include "operator_type.h"

#include "maparray_kernels.h"

namespace skepu
{


/*!
 *  A function called by the constructor. It creates the OpenCL program for the skeleton and saves a
 *  handle for the kernel. The program is built from a string containing the user function
 *  (specified when constructing the skeleton) and a generic MapArray kernel. The type and function
 *  names in the generic kernel are relpaced by user function specific code before it is compiled
 *  by the OpenCL JIT compiler.
 *
 *  Also handles the use of doubles automatically by including "#pragma OPENCL EXTENSION cl_khr_fp64: enable" if doubles
 *  are used.
 */
template <typename MapArrayFunc>
void MapArray<MapArrayFunc>::createOpenCLProgram()
{
   //Creates the sourcecode
   std::string totalSource;
   std::string kernelSource;
   std::string funcSource = m_mapArrayFunc->func_CL;
   std::string kernelName;

   std::string kernelMatrixName;

   if(m_mapArrayFunc->datatype_CL == "double" || m_mapArrayFunc->constype_CL == "double")
   {
      totalSource.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");
   }

   if(m_mapArrayFunc->funcType == ARRAY)
   {
      kernelSource = MapArrayKernel_CL;
      kernelName = "MapArrayKernel_" + m_mapArrayFunc->funcName_CL;
   }
   else if(m_mapArrayFunc->funcType == ARRAY_INDEX)
   {
      kernelSource = MapArrayKernel_CL_Matrix;
      kernelName = "MapArrayKernel_Matrix_" + m_mapArrayFunc->funcName_CL;
   }
   else if(m_mapArrayFunc->funcType == ARRAY_INDEX_BLOCK_WISE)
   {
      kernelSource = MapArrayKernel_CL_Matrix_Blockwise;
      kernelName = "MapArrayKernel_Matrix_Blockwise_" + m_mapArrayFunc->funcName_CL;
   }
   else if(m_mapArrayFunc->funcType == ARRAY_INDEX_SPARSE_BLOCK_WISE)
   {
      kernelSource = MapArrayKernel_CL_Sparse_Matrix_Blockwise;
      kernelName = "MapArrayKernel_Sparse_Matrix_Blockwise_" + m_mapArrayFunc->funcName_CL;
   }
   else
   {
      SKEPU_ERROR("Not valid function, should be ARRAY OR ARRAY_INDEX OR ARRAY_INDEX_BLOCK_WISE!\n");
   }

   // necessary for ARRAY_INDEX_BLOCK_WISE, ARRAY and ARRAY_INDEX
   replaceTextInString(kernelSource, std::string("CONST_TYPE"), m_mapArrayFunc->constype_CL);

   replaceTextInString(kernelSource, std::string("TYPE"), m_mapArrayFunc->datatype_CL);
   replaceTextInString(kernelSource, std::string("KERNELNAME"), m_mapArrayFunc->funcName_CL);
   replaceTextInString(kernelSource, std::string("FUNCTIONNAME"), m_mapArrayFunc->funcName_CL);
   
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
 *  Applies the MapArray skeleton to the two ranges of elements specified by iterators.
 *  Result is saved to a seperate output range. First range can be accessed entirely for each element in second range.
 *  The calculations can be performed by one or more devices. In the case of several, the input range is divided evenly
 *  among the participating devices. \em OpenCL is used as backend.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param numDevices Integer specifying the number of devices to perform the calculation on.
 */
template <typename MapArrayFunc>
template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
void MapArray<MapArrayFunc>::mapArrayNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, size_t numDevices)
{
   cl_int err;

   // Divide the elements amongst the devices
   size_t totalNumElements = input2End - input2Begin;
   size_t numElemPerSlice = totalNumElements / numDevices;
   size_t rest = totalNumElements % numDevices;
   size_t n1 = input1End - input1Begin;


   typename MapArrayFunc::CONST_TYPE const1 = m_mapArrayFunc->getConstant();

   typename Input1Iterator::device_pointer_type_cl in1_mem_p[MAX_GPU_DEVICES];
   typename Input2Iterator::device_pointer_type_cl in2_mem_p[MAX_GPU_DEVICES];
   typename OutputIterator::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   // First create OpenCL memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      in1_mem_p[i] = input1Begin.getParent().updateDevice_CL(input1Begin.getAddress(), n1, m_kernels_CL.at(i).second, false);
      in2_mem_p[i] = input2Begin.getParent().updateDevice_CL((input2Begin+i*numElemPerSlice).getAddress(), numElem, m_kernels_CL.at(i).second, false);

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
      in1_mem_p[i] = input1Begin.getParent().updateDevice_CL(input1Begin.getAddress(), n1, m_kernels_CL.at(i).second, true);
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
      clSetKernelArg(kernel, 4, sizeof(typename MapArrayFunc::CONST_TYPE), (void*)&const1);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!! " <<err <<"\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}

/*!
 *  Performs the MapArray on the two vectors with \em OpenCL as backend. Seperate output vector.
 *  First Vector can be accessed entirely for each element in second Vector.
 *  The function is a wrapper for
 *  CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CL(Vector<T>& input1, Vector<T>& input2, Vector<T>& output, int useNumGPU)
{
   if(input2.size() != output.size())
   {
      output.clear();
      output.resize(input2.size());
   }

   CL(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin(), useNumGPU);
}














/*!
 *  Performs MapArray on the one vector and one matrix block-wise with \em OpenCL as backend. Seperate output vector is used.
 *  The Vector can be accessed entirely for "a block of elements" in the Matrix. The block-length is specified in the user-function.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CL(Vector<T>& input1, SparseMatrix<T>& input2, Vector<T>& output, int useNumGPU)
{
   size_t nrows = input2.total_rows();
   size_t ncols = input2.total_cols();

   size_t p2BlockSize = m_mapArrayFunc->param2BlockSize;

   if( p2BlockSize!=ncols )
   {
      SKEPU_ERROR("For Vector-SparseMatrix MapArray operation: The 'p2BlockSize' specified in user function should be equal to the number of columns in the sparse matrix.\n");
   }

   size_t outSize = nrows; //nsize/p2BlockSize;

   if(outSize != output.size())
   {
      output.clear();
      output.resize(outSize);
   }

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }

   cl_int err;

   typename MapArrayFunc::CONST_TYPE const1;

   const1 = m_mapArrayFunc->getConstant();

   size_t n1 = input1.size();

   // Divide the elements amongst the devices
   size_t numElemPerSlice = outSize/numDevices;

   size_t restElem = outSize % numDevices;

   typename Vector<T>::device_pointer_type_cl in1_mem_p[MAX_GPU_DEVICES];
   typename SparseMatrix<T>::device_pointer_type_cl in2_mem_p[MAX_GPU_DEVICES];
   typename Vector<T>::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   typename SparseMatrix<T>::device_pointer_index_type_cl in2_row_offsets_mem_p[MAX_GPU_DEVICES];
   typename SparseMatrix<T>::device_pointer_index_type_cl in2_col_indices_mem_p[MAX_GPU_DEVICES];

   size_t offset = 0;
   size_t end = 0;

   // First create OpenCL memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+restElem;
      else
         numElem = numElemPerSlice;

      end = input2.get_rowOffsetFromStart(numElem+i*numElemPerSlice);

      in1_mem_p[i] = input1.updateDevice_CL(input1.getAddress(), n1, m_kernels_CL.at(i).second, false);
      in2_mem_p[i] = input2.updateDevice_CL((input2.get_values()+offset), end-offset, m_kernels_CL.at(i).second, false);

      in2_row_offsets_mem_p[i] = input2.updateDevice_Index_CL(((input2.get_row_pointers())+i*numElemPerSlice), numElem+1, m_kernels_CL.at(i).second, false);
      in2_col_indices_mem_p[i] = input2.updateDevice_Index_CL((input2.get_col_indices()+offset), end-offset, m_kernels_CL.at(i).second, false);

      out_mem_p[i] = output.updateDevice_CL(((output.getAddress())+i*numElemPerSlice), numElem, m_kernels_CL.at(i).second, false);

      offset = end;
   }

   offset = end = 0;

   for(size_t i = 0; i < numDevices; ++i)
   {
      // If there is a rest, last device takes care of it
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+restElem;
      else
         numElem = numElemPerSlice;

      end = input2.get_rowOffsetFromStart(numElem+i*numElemPerSlice);

      // Setup parameters
      BackEndParams bp=m_execPlan->find_(numElem);
      size_t maxBlocks = bp.maxBlocks;
      size_t maxThreads = bp.maxThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      localWorkSize[0] = std::min(maxThreads, (size_t)numElem);
      globalWorkSize[0] = (std::max((size_t)1, std::min( (numElem/(localWorkSize[0]) + (numElem%(localWorkSize[0]) == 0 ? 0:1)), maxBlocks))) * localWorkSize[0];

      // Copies the elements to the device
      in1_mem_p[i] = input1.updateDevice_CL(input1.getAddress(), n1, m_kernels_CL.at(i).second, true);
      in2_mem_p[i] = input2.updateDevice_CL((input2.get_values()+offset), end-offset, m_kernels_CL.at(i).second, true);

      in2_row_offsets_mem_p[i] = input2.updateDevice_Index_CL(((input2.get_row_pointers())+i*numElemPerSlice), numElem+1, m_kernels_CL.at(i).second, true);
      in2_col_indices_mem_p[i] = input2.updateDevice_Index_CL((input2.get_col_indices()+offset), end-offset, m_kernels_CL.at(i).second, true);

      cl_mem in1_p = in1_mem_p[i]->getDeviceDataPointer();
      cl_mem in2_p = in2_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_mem in2_row_offsets_p = in2_row_offsets_mem_p[i]->getDeviceDataPointer();
      cl_mem in2_col_indices_p = in2_col_indices_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&in2_row_offsets_p);
      clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&in2_col_indices_p);
      clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&offset);
      clSetKernelArg(kernel, 7, sizeof(typename MapArrayFunc::CONST_TYPE), (void*)&const1);

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!! " <<err <<"\n");
      }

      offset = end;

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}



/*!
 *  Performs MapArray on the one vector and one matrix block-wise with \em OpenCL as backend. Seperate output vector is used.
 *  The Vector can be accessed entirely for "a block of elements" in the Matrix. The block-length is specified in the user-function.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CL(Vector<T>& input1, Matrix<T>& input2, Vector<T>& output, int useNumGPU)
{
   size_t nrows = input2.total_rows();
   size_t ncols = input2.total_cols();

   size_t nsize = nrows * ncols;

   size_t p2BlockSize = m_mapArrayFunc->param2BlockSize;

   if( (nsize%p2BlockSize)!=0 )
   {
      SKEPU_ERROR("The 'p2BlockSize' specified in user function should be a perfect multiple of 'param2' size. Operation aborted.\n");
   }

   size_t outSize = nsize/p2BlockSize;

   if(outSize != output.size())
   {
      output.clear();
      output.resize(outSize);
   }

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }

   cl_int err;

   typename MapArrayFunc::CONST_TYPE const1;

   const1 = m_mapArrayFunc->getConstant();

   size_t n1 = input1.size();

   // Divide the elements amongst the devices
   size_t numElemPerSlice = outSize/numDevices;

   size_t restElem = outSize % numDevices;

   typename Vector<T>::device_pointer_type_cl in1_mem_p[MAX_GPU_DEVICES];
   typename Matrix<T>::device_pointer_type_cl in2_mem_p[MAX_GPU_DEVICES];
   typename Vector<T>::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   // First create OpenCL memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+restElem;
      else
         numElem = numElemPerSlice;

      in1_mem_p[i] = input1.updateDevice_CL(input1.getAddress(), n1, m_kernels_CL.at(i).second, false);
      in2_mem_p[i] = input2.updateDevice_CL(((input2.getAddress())+i*numElemPerSlice*p2BlockSize), numElem*p2BlockSize, m_kernels_CL.at(i).second, false);

      out_mem_p[i] = output.updateDevice_CL(((output.getAddress())+i*numElemPerSlice), numElem, m_kernels_CL.at(i).second, false);
   }

   for(size_t i = 0; i < numDevices; ++i)
   {
      // If there is a rest, last device takes care of it
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+restElem;
      else
         numElem = numElemPerSlice;

      // Setup parameters
      BackEndParams bp=m_execPlan->find_(numElem);
      size_t maxBlocks = bp.maxBlocks;
      size_t maxThreads = bp.maxThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      localWorkSize[0] = std::min(maxThreads, (size_t)numElem);
      globalWorkSize[0] = (std::max((size_t)1, std::min( (numElem/(localWorkSize[0]) + (numElem%(localWorkSize[0]) == 0 ? 0:1)), maxBlocks))) * localWorkSize[0];

      // Copies the elements to the device
      in1_mem_p[i] = input1.updateDevice_CL(input1.getAddress(), n1, m_kernels_CL.at(i).second, true);
      in2_mem_p[i] = input2.updateDevice_CL(((input2.getAddress())+i*numElemPerSlice*p2BlockSize), numElem*p2BlockSize, m_kernels_CL.at(i).second, true);

      cl_mem in1_p = in1_mem_p[i]->getDeviceDataPointer();
      cl_mem in2_p = in2_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&p2BlockSize);
      clSetKernelArg(kernel, 5, sizeof(typename MapArrayFunc::CONST_TYPE), (void*)&const1);

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!! " <<err <<"\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}


/*!
 *  Performs the MapArray on the one vector and one matrix with \em OpenCL as backend. Separate output matrix.
 *  The vector can be accessed entirely for each element in the matrix.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CL(Vector<T>& input1, Matrix<T>& input2, Matrix<T>& output, int useNumGPU)
{
   if(input2.size() != output.size())
   {
      output.clear();
      output.resize(input2.total_rows(), input2.total_cols());
   }

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }

   cl_int err;

   typename MapArrayFunc::CONST_TYPE const1;

   const1 = m_mapArrayFunc->getConstant();

   size_t nrows = input2.total_rows();
   size_t ncols = input2.total_cols();
   size_t n = nrows * ncols;
   size_t n1 = input1.size();

   // Divide the elements amongst the devices
   size_t numRowsPerSlice = nrows / numDevices;
   size_t numElemPerSlice = numRowsPerSlice * ncols;

   size_t restRows = nrows % numDevices;

   typename Vector<T>::device_pointer_type_cl in1_mem_p[MAX_GPU_DEVICES];
   typename Matrix<T>::device_pointer_type_cl in2_mem_p[MAX_GPU_DEVICES];
   typename Matrix<T>::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   // First create OpenCL memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem, numRows;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      numElem = numRows * ncols;

      in1_mem_p[i] = input1.updateDevice_CL(input1.getAddress(), n1, m_kernels_CL.at(i).second, false);
      in2_mem_p[i] = input2.updateDevice_CL(((input2.getAddress())+i*numElemPerSlice), numElem, m_kernels_CL.at(i).second, false);

      out_mem_p[i] = output.updateDevice_CL(((output.getAddress())+i*numElemPerSlice), numElem, m_kernels_CL.at(i).second, false);
   }

   size_t yoffset[MAX_GPU_DEVICES];
   size_t xsize[MAX_GPU_DEVICES];
   size_t ysize[MAX_GPU_DEVICES];

   for(size_t i = 0; i < numDevices; ++i)
   {
      // If there is a rest, last device takse care of it
      size_t numElem, numRows;
      if(i == numDevices-1)
         numRows = numRowsPerSlice+restRows;
      else
         numRows = numRowsPerSlice;

      numElem = numRows * ncols;

      // Setup parameters
      size_t globalWorkSize[2];
      size_t localWorkSize[2];
      localWorkSize[0] =  (ncols>32)? 32: ncols;                            // each thread does multiple Xs
      localWorkSize[1] =  (numRows>16)? 16: numRows;
      globalWorkSize[0] = ((ncols+(localWorkSize[0]-1)) / localWorkSize[0]) * localWorkSize[0];
      globalWorkSize[1] = ((numRows+(localWorkSize[1]-1))  / localWorkSize[1]) * localWorkSize[1];

      // Copies the elements to the device
      in1_mem_p[i] = input1.updateDevice_CL(input1.getAddress(), n1, m_kernels_CL.at(i).second, true);
      in2_mem_p[i] = input2.updateDevice_CL(((input2.getAddress())+i*numElemPerSlice), numElem, m_kernels_CL.at(i).second, true);

      cl_mem in1_p = in1_mem_p[i]->getDeviceDataPointer();
      cl_mem in2_p = in2_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      yoffset[i] = numRowsPerSlice*i;
      xsize[i] = ncols;
      ysize[i] = numRows;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&xsize[i]);
      clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&ysize[i]);
      clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&yoffset[i]);
      clSetKernelArg(kernel, 7, sizeof(typename MapArrayFunc::CONST_TYPE), (void*)&const1);

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
 *  Performs the MapArray on the two element ranges with \em  OpenCL as backend. Seperate output range.
 *  First range can be accessed entirely for each element in second range.
 *  Calls mapArrayNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, size_t numDevices).
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapArrayFunc>
template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
void MapArray<MapArrayFunc>::CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPARRAY OPENCL\n")

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   mapArrayNumDevices_CL(input1Begin, input1End, input2Begin, input2End, outputBegin, numDevices); // as here same method is invoked no matter how many GPUs we use
}



}

#endif

