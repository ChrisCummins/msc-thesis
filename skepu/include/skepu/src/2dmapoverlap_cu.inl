/*! \file 2dmapoverlap_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the MapOverlap2D skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>

#include "operator_type.h"

#include "mapoverlap_convol_kernels.h"
#include "device_mem_pointer_cu.h"
#include "device_cu.h"

namespace skepu
{


/*!
 *  Performs the 2D MapOverlap on a Matrix on the \em CUDA with the same Matrix as output.
 *  The actual filter is specified in a user-function.
 *
 *  \param input A Matrix that is used for both input and output.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::CU(Matrix<T>& input, int useNumGPU)
{
   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t filter_rows=m_mapOverlapFunc->overlapY;
   size_t filter_cols=m_mapOverlapFunc->overlapX;

   size_t out_rows=in_rows-(filter_rows*2);
   size_t out_cols=in_cols-(filter_cols*2);

   Matrix<T> output(out_rows,out_cols);

   CU(input, output, useNumGPU);

   output.updateHost();

   size_t k=0;
   for(size_t i= filter_rows; i<(out_rows+filter_rows); i++)
      for(size_t j=filter_cols; j<(out_cols+filter_cols); j++)
      {
         input(i*in_cols+j) = output(k++);
      }
}



/*!
 *  Performs the 2D MapOverlap using a single CUDA GPU.
 *  The actual filter is specified in a user-function.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param deviceID Integer specifying the which device to use.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::mapOverlapSingleThread_CU(Matrix<T>& input, Matrix<T>& output, unsigned int deviceID)
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

   cudaSetDevice(deviceID);

   typename Matrix<T>::device_pointer_type_cu in_mem_p = input.updateDevice_CU(input.GetArrayRep(), in_rows, in_cols, deviceID, true, false, true);
   typename Matrix<T>::device_pointer_type_cu out_mem_p = output.updateDevice_CU(output.GetArrayRep(), out_rows, out_cols, deviceID, false, true, true);

   dim3 numBlocks;
   dim3 numThreads;

   numThreads.x = (out_cols>16)? 16: out_cols;
   numThreads.y = (out_rows>32)? 32: out_rows;
   numThreads.z = 1;

   numBlocks.x = (out_cols + numThreads.x - 1) / numThreads.x;
   numBlocks.y = (out_rows + numThreads.y - 1) / numThreads.y;
   numBlocks.z = 1;

   m_mapOverlapFunc->setStride(numThreads.x + filter_cols-1);

   size_t sharedMem =  (numThreads.x + filter_cols-1) * (numThreads.y + filter_rows-1) * sizeof(T);

#ifdef USE_PINNED_MEMORY
   conv_cuda_2D_kernel<<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(deviceID)->m_streams[0]) >>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), out_rows, out_cols, filter_rows, filter_cols, in_mem_p->m_pitch, out_mem_p->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x + filter_cols-1) );
#else
   conv_cuda_2D_kernel<<< numBlocks,numThreads, sharedMem >>>(*m_mapOverlapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), out_rows, out_cols, filter_rows, filter_cols, in_mem_p->m_pitch, out_mem_p->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x + filter_cols-1) );
#endif

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}



/*!
 *  Performs the 2D MapOverlap using multiple CUDA GPUs.
 *  The actual filter is specified in a user-function.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param numDevices Integer specifying how many devices to use.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::mapOverlapMultipleThread_CU(Matrix<T>& input, Matrix<T>& output, size_t numDevices)
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
//    input.updateHostAndInvalidateDevice();

   typename Matrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
   typename Matrix<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

   // First create CUDA memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      cudaSetDevice(i);

      size_t outRows;
      if(i == numDevices-1)
         outRows = numRowsPerSlice+restRows;
      else
         outRows = numRowsPerSlice;

      size_t inRows = outRows+filter_rows-1; // no matter which device, number of input rows is same.

      in_mem_p[i] = input.updateDevice_CU(input.GetArrayRep()+i*numRowsPerSlice*in_cols, inRows, in_cols, i, false, false, true);

      out_mem_p[i] = output.updateDevice_CU(output.GetArrayRep()+i*numRowsPerSlice*out_cols, outRows, out_cols, i, false, false, true);
   }

   // Fill out argument struct with right information and start threads.
   for(size_t i = 0; i < numDevices; ++i)
   {
      cudaSetDevice(i);

      size_t outRows;
      if(i == numDevices-1)
         outRows = numRowsPerSlice+restRows;
      else
         outRows = numRowsPerSlice;

      size_t inRows = outRows+filter_rows-1; // no matter which device, number of input rows is same.

      in_mem_p[i] = input.updateDevice_CU(input.GetArrayRep()+i*numRowsPerSlice*in_cols, inRows, in_cols, i, true, false, true);

      out_mem_p[i] = output.updateDevice_CU(output.GetArrayRep()+i*numRowsPerSlice*out_cols, outRows, out_cols, i, false, true, true, true);

      dim3 numBlocks;
      dim3 numThreads;

      numThreads.x = (out_cols>16)? 16: out_cols;
      numThreads.y = (outRows>32)? 32: outRows;
      numThreads.z = 1;

      numBlocks.x = (out_cols + numThreads.x - 1) / numThreads.x;
      numBlocks.y = (outRows + numThreads.y - 1) / numThreads.y;
      numBlocks.z = 1;

      if(i==0) // only set first time
         m_mapOverlapFunc->setStride(numThreads.x + filter_cols-1);

      size_t sharedMem =  (numThreads.x + filter_cols-1) * (numThreads.y + filter_rows-1) * sizeof(T);

#ifdef USE_PINNED_MEMORY
      conv_cuda_2D_kernel<<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x + filter_cols-1) );
#else
      conv_cuda_2D_kernel<<< numBlocks,numThreads, sharedMem >>>(*m_mapOverlapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x + filter_cols-1) );
#endif

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}


/*!
 *  Performs the 2D MapOverlap on a whole matrix on the \em CUDA with a separate output matrix.
 *  The actual filter is specified in a user-function.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::CU(Matrix<T>& input, Matrix<T>& output, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("2D MAPOVERLAP CUDA\n")

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

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapOverlapSingleThread_CU(input, output, 0);
   }
   else
   {
      mapOverlapMultipleThread_CU(input, output, numDevices);
   }
}




/*!
 *  Performs the 2D MapOverlap on the \em CUDA, based on provided filter and input neighbouring elements on a whole Matrix.
 *  With a separate Matrix as output.
 *
 *  \param input A matrix which the mapping will be performed on. It should include padded data as well considering the filter size
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param filter The filter which will be applied for each element in the output.
 *  \param useTiling The boolean flag that specify whether to use tiling optimizations.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::CU(Matrix<T>& input, Matrix<T>& output, Matrix<T>& filter, bool useTiling, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("2D MAPOVERLAP with Filter Matrix CUDA\n")
   
   size_t numDevices = m_environment->m_devices_CU.size();

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

   size_t numRowsPerSlice = out_rows / numDevices;
   size_t restRows = out_rows % numDevices;

   //Need to get new values from other devices so that the overlap between devices is up to date.
   //Bad for performance since whole vector needs to be copied, fix so that only overlap is fetched and updated.
//    input.updateHostAndInvalidateDevice();

   typename Matrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
   typename Matrix<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

   // First create CUDA memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      cudaSetDevice(i);

      size_t outRows;
      if(i == numDevices-1)
         outRows = numRowsPerSlice+restRows;
      else
         outRows = numRowsPerSlice;

      size_t inRows = outRows+filter_rows-1;

      in_mem_p[i] = input.updateDevice_CU(input.GetArrayRep()+i*numRowsPerSlice*in_cols, inRows, in_cols, i, false, false, true);

      out_mem_p[i] = output.updateDevice_CU(output.GetArrayRep()+i*numRowsPerSlice*out_cols, outRows, out_cols, i, false, false, true);
   }

   //  copy data and do computation
   for(size_t i=0; i<numDevices; i++)
   {
      cudaSetDevice(i);

      size_t outRows;
      if(i == numDevices-1)
         outRows = numRowsPerSlice+restRows;
      else
         outRows = numRowsPerSlice;

      size_t inRows = outRows+filter_rows-1;

      in_mem_p[i] = input.updateDevice_CU(input.GetArrayRep()+i*numRowsPerSlice*in_cols, inRows, in_cols, i, true, false, true);
      out_mem_p[i] = output.updateDevice_CU(output.GetArrayRep()+i*numRowsPerSlice*out_cols, outRows, out_cols, i, false, true, true, true);

      // here 0 is offset and not the device id
      CHECK_CUDA_ERROR(cudaMemcpyToSymbol(deviceFilter, filter.GetArrayRep(), sizeof(T)*(filter_rows*filter_cols), 0, cudaMemcpyHostToDevice));


      dim3 numBlocks;
      dim3 numThreads;

      numThreads.x = (out_cols>16)? 16: out_cols;
      numThreads.y = (out_rows>32)? 32: out_rows;
      numThreads.z = 1;

      size_t tilingFac = 1;

      if(useTiling)
         tilingFac=calculateTiling<int>(18, filter_cols, filter_rows, out_cols);

      if (tilingFac>16) // cannot do more tiling than 16
         tilingFac = 16;

      numBlocks.x = (out_cols + (numThreads.x*tilingFac) - 1) / (numThreads.x * tilingFac);
      numBlocks.y = (out_rows + numThreads.y - 1) / numThreads.y;
      numBlocks.z = 1;

      size_t sharedMem =  (numThreads.x * tilingFac + filter_cols-1) * (numThreads.y + filter_rows-1) * sizeof(T);

      if(useTiling)
      {
         if(tilingFac==16)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_16_kernel<true><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_16_kernel<true><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==14)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_14_kernel<true><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_14_kernel<true><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==12)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_12_kernel<true><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_12_kernel<true><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==10)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_10_kernel<true><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_10_kernel<true><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==8)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_8_kernel<true><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_8_kernel<true><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==6)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_6_kernel<true><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_6_kernel<true><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==4)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_4_kernel<true><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_4_kernel<true><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==2)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_2_kernel<true><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_2_kernel<true><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_kernel<true><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), inRows, in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_kernel<true><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), inRows, in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
      }
      else
      {
#ifdef USE_PINNED_MEMORY
         conv_cuda_shared_kernel<true><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), inRows, in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
         conv_cuda_shared_kernel<true><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), inRows, in_cols, outRows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();

   }// end for-loop
}



/*!
 *  Performs the 2D MapOverlap on the \em CUDA, by taking average of neighbouring elements on a whole Matrix.
 *  With a separate Matrix as output.
 *
 *  \param input A matrix which the mapping will be performed on. It should include padded data as well considering the filter size
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param filter_rows The number of rows used as neighbouring elements to calculate new value for each output element.
 *  \param filter_cols The number of columns used as neighbouring elements to calculate new value for each output element.
 *  \param useTiling The boolean flag that specify whether to use tiling optimizations.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::CU(Matrix<T>& input, Matrix<T>& output, size_t filter_rows, size_t filter_cols, bool useTiling, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("2D MAPOVERLAP with Average Filter CUDA\n")
   
   size_t numDevices = m_environment->m_devices_CU.size();

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

   typename Matrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
   typename Matrix<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

   // First create CUDA memory if not created already.
   for(size_t i=0; i<numDevices; i++)
   {
      if(i==numDevices-1)
      {
         in_rows= in_rows+rest_rows;
         out_rows= out_rows+rest_rows;
      }

      // Copy elements to device and allocate output memory.
      in_mem_p[i] = input.updateDevice_CU(input.GetArrayRep()+(i*out_rows*in_cols), in_rows, in_cols, i, false /*!copy*/, false /*!writeAccess*/, true /*!usePitch*/);

      out_mem_p[i] = output.updateDevice_CU(output.GetArrayRep()+(i*out_rows*out_cols), out_rows, out_cols, i, false, false, true);
   }

   //  copy data and do computation
   for(size_t i=0; i<numDevices; i++)
   {
      cudaSetDevice(i);

      if(i==numDevices-1)
      {
         in_rows= in_rows+rest_rows;
         out_rows= out_rows+rest_rows;
      }

      in_mem_p[i] = input.updateDevice_CU(input.GetArrayRep()+(i*out_rows*in_cols), in_rows, in_cols, i, true, false, true);
      out_mem_p[i] = output.updateDevice_CU(output.GetArrayRep()+(i*out_rows*out_cols), out_rows, out_cols, i, false, true, true, true);

      dim3 numBlocks;
      dim3 numThreads;

      numThreads.x = (out_cols>16)? 16: out_cols;
      numThreads.y = (out_rows>32)? 32: out_rows;
      numThreads.z = 1;

      size_t tilingFac = 1;

      if(useTiling)
         tilingFac=calculateTiling<int>(14, filter_cols, filter_rows, out_cols);

      if (tilingFac>16) // cannot do more tiling than 16
         tilingFac = 16;

      numBlocks.x = (out_cols + (numThreads.x*tilingFac) - 1) / (numThreads.x * tilingFac);
      numBlocks.y = (out_rows + numThreads.y - 1) / numThreads.y;
      numBlocks.z = 1;

      size_t sharedMem =  (numThreads.x * tilingFac + filter_cols-1) * (numThreads.y + filter_rows-1) * sizeof(T);

      if(useTiling)
      {
         if(tilingFac==16)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_16_kernel<false><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_16_kernel<false><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==14)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_14_kernel<false><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_14_kernel<false><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==12)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_12_kernel<false><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_12_kernel<false><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==10)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_10_kernel<false><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_10_kernel<false><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==8)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_8_kernel<false><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_8_kernel<false><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==6)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_6_kernel<false><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_6_kernel<false><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==4)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_4_kernel<false><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_4_kernel<false><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else if(tilingFac==2)
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_tiling_2_kernel<false><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_tiling_2_kernel<false><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
         else
         {
#ifdef USE_PINNED_MEMORY
            conv_cuda_shared_kernel<false><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_rows, in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
            conv_cuda_shared_kernel<false><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_rows, in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
         }
      }
      else
      {
#ifdef USE_PINNED_MEMORY
         conv_cuda_shared_kernel<false><<< numBlocks,numThreads, sharedMem, (m_environment->m_devices_CU.at(i)->m_streams[0]) >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_rows, in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#else
         conv_cuda_shared_kernel<false><<< numBlocks,numThreads, sharedMem >>>(in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), in_rows, in_cols, out_rows, out_cols, filter_rows, filter_cols, in_mem_p[i]->m_pitch, out_mem_p[i]->m_pitch,  (numThreads.y + filter_rows-1), (numThreads.x * tilingFac + filter_cols-1) );
#endif
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}

}
#endif


