/*! \file skepu_opencl_helpers.h
 *  \brief Contains the definitions of some helper functions related to \em OpenCL backend.
 */

#ifndef SKEPU_CUDA_HELPER_H
#define SKEPU_CUDA_HELPER_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "device_cl.h"


namespace skepu
{


#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif


// Give a little more for Windows : the console window often disapears before we can read the message
#ifdef _WIN32
# if 1//ndef UNICODE
#  ifdef _DEBUG // Do this only in debug mode...
inline void VSPrintf(FILE *file, LPCSTR fmt, ...)
{
   size_t fmt2_sz	= 2048;
   char *fmt2		= (char*)malloc(fmt2_sz);
   va_list  vlist;
   va_start(vlist, fmt);
   while((_vsnprintf(fmt2, fmt2_sz, fmt, vlist)) < 0) // means there wasn't anough room
   {
      fmt2_sz *= 2;
      if(fmt2) free(fmt2);
      fmt2 = (char*)malloc(fmt2_sz);
   }
   OutputDebugStringA(fmt2);
   fprintf(file, fmt2);
   free(fmt2);
}
#	define FPRINTF(a) VSPrintf a
#  else //debug
#	define FPRINTF(a) fprintf a
// For other than Win32
#  endif //debug
# else //unicode
// Unicode case... let's give-up for now and keep basic printf
#	define FPRINTF(a) fprintf a
# endif //unicode
#else //win32
#	define FPRINTF(a) fprintf a
#endif //win32


template <typename T>
void copyDeviceToHost(T *hostPtr, cl_mem devPtr, size_t numElements, Device_CL* device, size_t offset)
{
   if(devPtr != NULL && hostPtr != NULL)
   {
      DEBUG_TEXT_LEVEL2("** DEVICE_TO_HOST OpenCL: "<< numElements <<"!!!\n")

      cl_int err;

      size_t sizeVec;

      sizeVec = numElements*sizeof(T);

      err = clEnqueueReadBuffer(device->getQueue(), devPtr, CL_TRUE, offset, sizeVec, (void*)hostPtr, 0, NULL, NULL);

      if(err != CL_SUCCESS)
      {
         FPRINTF((stderr, "Error copying data from device\n"));
      }
   }
}



template <typename T>
void copyHostToDevice(T *hostPtr, cl_mem devPtr, size_t numElements, Device_CL* device, size_t offset)
{
   if(hostPtr != NULL && devPtr != NULL)
   {
      DEBUG_TEXT_LEVEL2("** HOST_TO_DEVICE OpenCL: "<< numElements <<"!!!\n")

      cl_int err;

      size_t sizeVec;

      sizeVec = numElements*sizeof(T);

      err = clEnqueueWriteBuffer(device->getQueue(), devPtr, CL_TRUE, offset, sizeVec, (void*)hostPtr, 0, NULL, NULL);

      if(err != CL_SUCCESS)
      {
         FPRINTF((stderr, "Error copying data to device\n"));
      }
   }
}


template <typename T>
inline cl_mem allocateOpenCLMemory(size_t size, Device_CL* device)
{
   DEBUG_TEXT_LEVEL2("** ALLOC OpenCL: "<< size <<"!!!\n")

   cl_int err;
   cl_mem devicePointer;

   size_t sizeVec = size*sizeof(T);

   devicePointer = clCreateBuffer(device->getContext(), CL_MEM_READ_WRITE, sizeVec, NULL, &err);
   if(err != CL_SUCCESS)
   {
      FPRINTF((stderr, "Error allocating memory on device\n"));
   }

   return devicePointer;
}


template <typename T>
inline void freeOpenCLMemory(cl_mem d_pointer)
{
   DEBUG_TEXT_LEVEL2("** DE-ALLOC OpenCL !!!\n")

//	if(d_pointer!=NULL)
   {
      if(clReleaseMemObject(d_pointer) != CL_SUCCESS)
         FPRINTF((stderr, "Error releasing memory on device\n"));
   }
}



/*!
 *  A helper function that is used to call the actual kernel for reduction. Used by other functions to call the actual kernel
 *  Internally, it just calls 2 kernels by setting their arguments. No synchronization is enforced.
 *
 *  \param n size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param in_p OpenCL memory pointer to input array.
 *  \param out_p OpenCL memory pointer to output array.
 *  \param kernel OpenCL kernel handle.
 *  \param device OpenCL device handle.
 */
template <typename T>
void ExecuteReduceOnADevice(size_t  n, const size_t  &numThreads, const size_t  &numBlocks, _cl_mem*& in_p, _cl_mem*& out_p, cl_kernel &kernel, Device_CL *device)
{
   cl_int err;

   size_t globalWorkSize[1];
   size_t localWorkSize[1];

   size_t sharedMemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);

   // Sets the kernel arguments for first reduction
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&n);
   clSetKernelArg(kernel, 3, sharedMemSize, NULL);

   globalWorkSize[0] = numBlocks * numThreads;
   localWorkSize[0] = numThreads;

   // First reduce all elements blockwise so that each block produces one element.
   err = clEnqueueNDRangeKernel(device->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      std::cerr<<"Error launching kernel RowWise!! 1st\n";
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
   err = clEnqueueNDRangeKernel(device->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      std::cerr<<"Error launching kernel RowWise!! 2nd\n";
   }
}




/*!
 *  A helper function used by createOpenCLProgram(). It finds all instances of a string in another string and replaces it with
 *  a third string.
 *
 *  \param text A \p std::string which is searched.
 *  \param find The \p std::string which is searched for and replaced.
 *  \param replace The relpacement \p std::string.
 */
void replaceTextInString(std::string& text, std::string find, std::string replace)
{
   std::string::size_type pos=0;
   while((pos = text.find(find, pos)) != std::string::npos)
   {
      text.erase(pos, find.length());
      text.insert(pos, replace);
      pos+=replace.length();
   }
}


}

#endif
