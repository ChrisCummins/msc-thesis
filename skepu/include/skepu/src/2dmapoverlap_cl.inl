/*! \file 2dmapoverlap_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the MapOverlap2D skeleton.
 */

#ifdef SKEPU_OPENCL

#include <iostream>

#include "operator_type.h"

#include "mapoverlap_convol_kernels.h"

#include "device_mem_pointer_cl.h"

namespace skepu
{

/*!
 *  A helper function used by createOpenCLProgram(). It finds all instances of a string in another string and replaces it with
 *  a third string.
 *
 *  \param text A \p std::string which is searched.
 *  \param find The \p std::string which is searched for and replaced.
 *  \param replace The relpacement \p std::string.
 */
template <typename MapOverlap2DFunc>
void MapOverlap2D<MapOverlap2DFunc>::replaceText(std::string& text, std::string find, std::string replace)
{
   std::string::size_type pos=0;
   while((pos = text.find(find, pos)) != std::string::npos)
   {
      text.erase(pos, find.length());
      text.insert(pos, replace);
      pos+=replace.length();
   }
}

/*!
 *  A function called by the constructor. It creates the OpenCL program for the skeleton and saves a handle for
 *  the kernel. The program is built from a string containing the user function (specified when constructing the
 *  skeleton) and a generic MapOverlap2D kernel. The type and function names in the generic kernel are relpaced by user function
 *  specific code before it is compiled by the OpenCL JIT compiler.
 *
 *  Also handles the use of doubles automatically by including "#pragma OPENCL EXTENSION cl_khr_fp64: enable" if doubles
 *  are used.
 */
template <typename MapOverlap2DFunc>
void MapOverlap2D<MapOverlap2DFunc>::createOpenCLProgram()
{
   //Creates the sourcecode
   std::string totalSource;
   std::string kernelSource;


   std::string funcSource = m_mapOverlapFunc->func_CL;
   std::string kernelName_2D, kernelName_ConvolFilter, kernelName_Convol;

   if(m_mapOverlapFunc->datatype_CL == "double")
   {
      totalSource.append("#pragma OPENCL EXTENSION cl_khr_fp64: enable\n");
   }

   kernelSource = MatrixConvol2D_CL +  MatrixConvolSharedFilter_CL + MatrixConvolShared_CL;

   kernelName_2D = "conv_opencl_2D_" + m_mapOverlapFunc->funcName_CL;

   kernelName_ConvolFilter = "conv_opencl_shared_filter_" + m_mapOverlapFunc->funcName_CL;

   kernelName_Convol = "conv_opencl_shared_" + m_mapOverlapFunc->funcName_CL;
   
   replaceText(kernelSource, std::string("TYPE"), m_mapOverlapFunc->datatype_CL);
   replaceText(kernelSource, std::string("KERNELNAME"), m_mapOverlapFunc->funcName_CL);
   replaceText(kernelSource, std::string("FUNCTIONNAME"), m_mapOverlapFunc->funcName_CL);
   
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

      temp_kernel = clCreateKernel(temp_program, kernelName_2D.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<kernelName_2D.c_str() <<"\n" <<err <<"\n");
      }
      m_kernels_2D_CL.push_back(std::make_pair(temp_kernel, (*it)));

      temp_kernel = clCreateKernel(temp_program, kernelName_ConvolFilter.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<kernelName_ConvolFilter.c_str() <<"\n" <<err <<"\n");
      }
      m_kernels_Mat_ConvolFilter_CL.push_back(std::make_pair(temp_kernel, (*it)));

      temp_kernel = clCreateKernel(temp_program, kernelName_Convol.c_str(), &err);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error creating a kernel!!\n" <<kernelName_Convol.c_str() <<"\n" <<err <<"\n");
      }
      m_kernels_Mat_Convol_CL.push_back(std::make_pair(temp_kernel, (*it)));
   }
}

/*!
 *  Performs the 2D MapOverlap on a Matrix on the \em OpenCL with the same Matrix as output.
 *  The actual filter is specified in a user-function.
 *
 *  \param input A Matrix that is used for both input and output.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::CL(Matrix<T>& input, int useNumGPU)
{
   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t filter_rows=m_mapOverlapFunc->overlapY;
   size_t filter_cols=m_mapOverlapFunc->overlapX;

   size_t out_rows=in_rows-(filter_rows*2);
   size_t out_cols=in_cols-(filter_cols*2);

   Matrix<T> output(out_rows,out_cols);

   CL(input, output, useNumGPU);

   output.updateHost();

   size_t k=0;
   for(size_t i= filter_rows; i<(out_rows+filter_rows); i++)
      for(size_t j=filter_cols; j<(out_cols+filter_cols); j++)
      {
         input(i*in_cols+j) = output(k++);
      }
}



/*!
 *  Performs the 2D MapOverlap using a single OpenCL GPU.
 *  The actual filter is specified in a user-function.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param deviceID Integer specifying the which device to use.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::mapOverlapSingleThread_CL(Matrix<T>& input, Matrix<T>& output, unsigned int deviceID)
{
   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t out_rows=output.total_rows();
   size_t out_cols=output.total_cols();

   size_t filter_rows=m_mapOverlapFunc->overlapY;
   size_t filter_cols=m_mapOverlapFunc->overlapX;

   //change filter_rows filter_cols to a different representation, used internally in this implementation
   filter_rows = filter_rows*2+1;
   filter_cols = filter_cols*2+1;

   typename Matrix<T>::device_pointer_type_cl in_mem_p = input.updateDevice_CL(input.GetArrayRep(), in_rows, in_cols, m_kernels_2D_CL.at(0).second, true);
   typename Matrix<T>::device_pointer_type_cl out_mem_p = output.updateDevice_CL(output.GetArrayRep(), out_rows, out_cols, m_kernels_2D_CL.at(0).second, false);

   size_t numBlocks[3];
   size_t numThreads[3];

   numThreads[0] = (out_cols>16)? 16: out_cols;
   numThreads[1] = (out_rows>32)? 32: out_rows;
   numThreads[2] = 1;

   numBlocks[0] = ( (size_t) ((out_cols + numThreads[0] - 1) / numThreads[0]) ) * numThreads[0];
   numBlocks[1] = ( (size_t) ((out_rows + numThreads[1] - 1) / numThreads[1]) ) * numThreads[1];
   numBlocks[2] = 1;

   size_t sharedRows = (numThreads[1] + filter_rows-1);
   size_t sharedCols = (numThreads[0] + filter_cols-1);
   size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);

   cl_mem in_p = in_mem_p->getDeviceDataPointer();
   cl_mem out_p = out_mem_p->getDeviceDataPointer();

   cl_kernel kernel = m_kernels_2D_CL.at(0).first;

   size_t stride = numThreads[0] + filter_cols-1;

   // Sets the kernel arguments
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&out_rows);
   clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&out_cols);
   clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&filter_rows);
   clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&filter_cols);
   clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&in_cols); //in_pitch
   clSetKernelArg(kernel, 7, sizeof(size_t), (void*)&out_cols); //out_pitch
   clSetKernelArg(kernel, 8, sizeof(size_t), (void*)&stride);
   clSetKernelArg(kernel, 9, sizeof(size_t), (void*)&sharedRows);
   clSetKernelArg(kernel,10, sizeof(size_t), (void*)&sharedCols);
   clSetKernelArg(kernel,11, sharedMemSize, NULL);


   // Launches the kernel (asynchronous)
   cl_int err = clEnqueueNDRangeKernel(m_kernels_2D_CL.at(0).second->getQueue(), kernel, 2, NULL, numBlocks, numThreads, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error launching MapOverlap2D kernel!! " <<err <<"\n");
   }

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}



/*!
 *  Performs the 2D MapOverlap using multiple OpenCL GPUs.
 *  The actual filter is specified in a user-function.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param numDevices Integer specifying how many devices to use.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::mapOverlapMultipleThread_CL(Matrix<T>& input, Matrix<T>& output, size_t numDevices)
{
   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t out_rows=output.total_rows();
   size_t out_cols=output.total_cols();

   size_t filter_rows=m_mapOverlapFunc->overlapY;
   size_t filter_cols=m_mapOverlapFunc->overlapX;

   //change filter_rows filter_cols to a different representation, used internally in this implementation
   filter_rows = filter_rows*2+1;
   filter_cols = filter_cols*2+1;

   size_t numRowsPerSlice = out_rows / numDevices;
   size_t restRows = out_rows % numDevices;

   //Need to get new values from other devices so that the overlap between devices is up to date.
   //Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
   input.updateHostAndInvalidateDevice();

   size_t numBlocks[3];
   size_t numThreads[3];

   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t outRows;
      if(i == numDevices-1)
         outRows = numRowsPerSlice+restRows;
      else
         outRows = numRowsPerSlice;

      size_t inRows = outRows+2*filter_rows; // no matter which device, number of input rows is same.

      typename Matrix<T>::device_pointer_type_cl in_mem_p = input.updateDevice_CL(input.GetArrayRep()+i*numRowsPerSlice*in_cols, inRows, in_cols, m_kernels_2D_CL.at(i).second, true);
      typename Matrix<T>::device_pointer_type_cl out_mem_p = output.updateDevice_CL(output.GetArrayRep()+i*numRowsPerSlice*out_cols, outRows, out_cols, m_kernels_2D_CL.at(i).second, false);

      numThreads[0] = (out_cols>16)? 16: out_cols;
      numThreads[1] = (outRows>32)? 32: outRows;
      numThreads[2] = 1;

      numBlocks[0] = ( (size_t) ((out_cols + numThreads[0] - 1) / numThreads[0]) ) * numThreads[0];
      numBlocks[1] = ( (size_t) ((outRows + numThreads[1] - 1) / numThreads[1]) ) * numThreads[1];
      numBlocks[2] = 1;

      size_t sharedRows = (numThreads[1] + filter_rows-1);
      size_t sharedCols = (numThreads[0] + filter_cols-1);
      size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);

      cl_mem in_p = in_mem_p->getDeviceDataPointer();
      cl_mem out_p = out_mem_p->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_2D_CL.at(i).first;

      size_t stride = numThreads[0] + filter_cols-1;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&outRows);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&out_cols);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&filter_rows);
      clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&filter_cols);
      clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&in_cols); //in_pitch
      clSetKernelArg(kernel, 7, sizeof(size_t), (void*)&out_cols); //out_pitch
      clSetKernelArg(kernel, 8, sizeof(size_t), (void*)&stride);
      clSetKernelArg(kernel, 9, sizeof(size_t), (void*)&sharedRows);
      clSetKernelArg(kernel,10, sizeof(size_t), (void*)&sharedCols);
      clSetKernelArg(kernel,11, sharedMemSize, NULL);


      // Launches the kernel (asynchronous)
      cl_int err = clEnqueueNDRangeKernel(m_kernels_2D_CL.at(i).second->getQueue(), kernel, 2, NULL, numBlocks, numThreads, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching MapOverlap2D kernel!! " <<err <<"\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p->changeDeviceData();
   }


}


/*!
 *  Performs the 2D MapOverlap on a whole matrix on the \em OpenCL with a separate output matrix.
 *  The actual filter is specified in a user-function.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::CL(Matrix<T>& input, Matrix<T>& output, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("2D MAPOVERLAP OpenCL\n")

   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t out_rows=output.total_rows();
   size_t out_cols=output.total_cols();

   size_t filter_rows=m_mapOverlapFunc->overlapY;
   size_t filter_cols=m_mapOverlapFunc->overlapX;

   if( ( (in_rows-(filter_rows*2)) != out_rows) && ( (in_cols-(filter_cols*2)) != out_cols))
   {
      output.clear();
      output.resize(in_rows-(filter_rows*2), in_cols-(filter_cols*2));
   }

   size_t numDevices = m_kernels_2D_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapOverlapSingleThread_CL(input, output, 0);
   }
   else
   {
      mapOverlapMultipleThread_CL(input, output, numDevices);
   }
}






/*!
 *  Performs the 2D MapOverlap on the \em OpenCL, based on provided filter and input neighbouring elements on a whole Matrix
 *  With a separate Matrix as output.
 *
 *  \param input A matrix which the mapping will be performed on. It should include padded data as well considering the filter size
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param filter The filter which will be applied for each element in the output.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::CL(Matrix<T>& input, Matrix<T>& output, Matrix<T>& filter, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("2D MAPOVERLAP with Filter Matrix OpenCL\n")
   
   size_t numDevices = m_kernels_Mat_ConvolFilter_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }

   //change filter_rows filter_cols to a different representation, used internally in this implementation
   size_t filter_rows=filter.total_rows();
   size_t filter_cols=filter.total_cols();

   // get lengths etc...
   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t out_rows=output.total_rows();
   size_t out_cols=output.total_cols();

   if( ( (in_rows-filter_rows+1) != out_rows) && ( (in_cols-filter_cols+1) != out_cols))
   {
      output.clear();
      output.resize((in_rows-filter_rows+1), (in_cols-filter_cols+1));
   }

   out_rows= (numDevices==1) ? out_rows : out_rows/numDevices; // out_rows per device, dividing the work evenly
   size_t rest_rows = (numDevices==1) ? 0 : (output.total_rows())%numDevices; // extra work
   in_rows = (numDevices==1) ? in_rows  : out_rows + filter_rows-1; // in_rows per device, dividing the work

   for(size_t dev=0; dev<numDevices; dev++)
   {
      if(dev==numDevices-1)
      {
         in_rows= in_rows+rest_rows;
         out_rows= out_rows+rest_rows;
      }

      typename Matrix<T>::device_pointer_type_cl in_mem_p = input.updateDevice_CL(input.GetArrayRep()+(dev*out_rows*in_cols), in_rows, in_cols, m_kernels_Mat_ConvolFilter_CL.at(dev).second, true);
      typename Matrix<T>::device_pointer_type_cl out_mem_p = output.updateDevice_CL(output.GetArrayRep()+(dev*out_rows*out_cols), out_rows, out_cols, m_kernels_Mat_ConvolFilter_CL.at(dev).second, false);

      typename Matrix<T>::device_pointer_type_cl filter_mem_p = filter.updateDevice_CL(filter.GetArrayRep(), filter_rows, filter_cols, m_kernels_Mat_ConvolFilter_CL.at(dev).second, true);

      size_t numBlocks[3];
      size_t numThreads[3];

      numThreads[0] = (out_cols>16)? 16: out_cols;
      numThreads[1] = (out_rows>32)? 32: out_rows;
      numThreads[2] = 1;

      numBlocks[0] = ( (size_t) ((out_cols + numThreads[0] - 1) / numThreads[0]) ) * numThreads[0];
      numBlocks[1] = ( (size_t) ((out_rows + numThreads[1] - 1) / numThreads[1]) ) * numThreads[1];
      numBlocks[2] = 1;

      size_t sharedRows = (numThreads[1] + filter_rows-1);
      size_t sharedCols = (numThreads[0] + filter_cols-1);
      size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);

      cl_mem in_p = in_mem_p->getDeviceDataPointer();
      cl_mem out_p = out_mem_p->getDeviceDataPointer();
      cl_mem filter_p = filter_mem_p->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_Mat_ConvolFilter_CL.at(dev).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&filter_p);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&in_rows);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&in_cols);
      clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&out_rows);
      clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&out_cols);
      clSetKernelArg(kernel, 7, sizeof(size_t), (void*)&filter_rows);
      clSetKernelArg(kernel, 8, sizeof(size_t), (void*)&filter_cols);
      clSetKernelArg(kernel, 9, sizeof(size_t), (void*)&in_cols); //in_pitch
      clSetKernelArg(kernel,10, sizeof(size_t), (void*)&out_cols); //out_pitch
      clSetKernelArg(kernel,11, sizeof(size_t), (void*)&sharedRows);
      clSetKernelArg(kernel,12, sizeof(size_t), (void*)&sharedCols);
      clSetKernelArg(kernel, 13, sharedMemSize, NULL);


      // Launches the kernel (asynchronous)
      cl_int err = clEnqueueNDRangeKernel(m_kernels_Mat_ConvolFilter_CL.at(dev).second->getQueue(), kernel, 2, NULL, numBlocks, numThreads, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching MapOverlap2D kernel!! " <<err <<"\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p->changeDeviceData();
   }
}



/*!
 *  Performs the 2D MapOverlap on the \em OpenCL, by taking average of neighbouring elements on a whole Matrix.
 *  With a separate Matrix as output.
 *
 *  \param input A matrix which the mapping will be performed on. It should include padded data as well considering the filter size
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param filter_rows The number of rows used as neighbouring elements to calculate new value for each output element.
 *  \param filter_cols The number of columns used as neighbouring elements to calculate new value for each output element.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::CL(Matrix<T>& input,Matrix<T>& output, size_t filter_rows, size_t filter_cols, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("2D MAPOVERLAP with Average Filter OpenCL\n")
   
   size_t numDevices = m_kernels_Mat_Convol_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }

   //change filter_rows filter_cols to a different representation, used internally in this implementation
   filter_rows = filter_rows*2+1;
   filter_cols = filter_cols*2+1;

   // get lengths etc...
   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t out_rows=output.total_rows();
   size_t out_cols=output.total_cols();

   if( ( (in_rows-filter_rows+1) != out_rows) && ( (in_cols-filter_cols+1) != out_cols))
   {
      output.clear();
      output.resize((in_rows-filter_rows+1), (in_cols-filter_cols+1));
   }

   out_rows= (numDevices==1) ? out_rows : out_rows/numDevices; // out_rows per device, dividing the work evenly
   size_t rest_rows = (numDevices==1) ? 0 : (output.total_rows())%numDevices; // extra work
   in_rows = (numDevices==1) ? in_rows  : out_rows + filter_rows-1; // in_rows per device, dividing the work

   for(size_t dev=0; dev<numDevices; dev++)
   {
      if(dev==numDevices-1)
      {
         in_rows= in_rows+rest_rows;
         out_rows= out_rows+rest_rows;
      }

      typename Matrix<T>::device_pointer_type_cl in_mem_p = input.updateDevice_CL(input.GetArrayRep()+(dev*out_rows*in_cols), in_rows, in_cols, m_kernels_Mat_Convol_CL.at(dev).second, true);
      typename Matrix<T>::device_pointer_type_cl out_mem_p = output.updateDevice_CL(output.GetArrayRep()+(dev*out_rows*out_cols), out_rows, out_cols, m_kernels_Mat_Convol_CL.at(dev).second, false);

      size_t numBlocks[3];
      size_t numThreads[3];

      numThreads[0] = (out_cols>16)? 16: out_cols;
      numThreads[1] = (out_rows>32)? 32: out_rows;
      numThreads[2] = 1;

      numBlocks[0] = ( (size_t) ((out_cols + numThreads[0] - 1) / numThreads[0]) ) * numThreads[0];
      numBlocks[1] = ( (size_t) ((out_rows + numThreads[1] - 1) / numThreads[1]) ) * numThreads[1];
      numBlocks[2] = 1;

      size_t sharedRows = (numThreads[1] + filter_rows-1);
      size_t sharedCols = (numThreads[0] + filter_cols-1);
      size_t sharedMemSize =  sharedRows * sharedCols * sizeof(T);

      cl_mem in_p = in_mem_p->getDeviceDataPointer();
      cl_mem out_p = out_mem_p->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_Mat_Convol_CL.at(dev).first;


      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&in_rows);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&in_cols);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&out_rows);
      clSetKernelArg(kernel, 5, sizeof(size_t), (void*)&out_cols);
      clSetKernelArg(kernel, 6, sizeof(size_t), (void*)&filter_rows);
      clSetKernelArg(kernel, 7, sizeof(size_t), (void*)&filter_cols);
      clSetKernelArg(kernel, 8, sizeof(size_t), (void*)&in_cols); //in_pitch
      clSetKernelArg(kernel, 9, sizeof(size_t), (void*)&out_cols); //out_pitch
      clSetKernelArg(kernel,10, sizeof(size_t), (void*)&sharedRows);
      clSetKernelArg(kernel,11, sizeof(size_t), (void*)&sharedCols);
      clSetKernelArg(kernel, 12, sharedMemSize, NULL);


      // Launches the kernel (asynchronous)
      cl_int err = clEnqueueNDRangeKernel(m_kernels_Mat_Convol_CL.at(dev).second->getQueue(), kernel, 2, NULL, numBlocks, numThreads, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching MapOverlap2D kernel!! " <<err <<"\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p->changeDeviceData();
   }
}



}
#endif


