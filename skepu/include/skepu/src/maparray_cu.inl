/*! \file maparray_cu.inl
 *  \brief Contains the definitions of CUDA specific member functions for the MapArray skeleton.
 */

#ifdef SKEPU_CUDA

#include <cuda.h>
#include <iostream>

#include "operator_type.h"

#include "maparray_kernels.h"
#include "device_mem_pointer_cu.h"
#include "device_cu.h"

namespace skepu
{

/*!
 *  Applies the MapArray skeleton to the two ranges of elements specified by iterators.
 *  Result is saved to a seperate output range. First range can be accessed entirely for each element in second range.
 *  The calculations are performed by one host thread using \p one device with \em CUDA as backend.
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
void MapArray<MapArrayFunc>::mapArraySingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, unsigned int deviceID)
{
   CHECK_CUDA_ERROR(cudaSetDevice(deviceID));
//    std::cerr << "%%%% MapArray GPU_" << deviceID << ", best: " << m_environment->bestCUDADevID << "\n\n";
   
   // Setup parameters
   size_t n = input2End-input2Begin;
   size_t n1 = input1End-input1Begin;

   BackEndParams bp = m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks; //Chosen somewhat arbitrarly
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;

   numThreads = std::min(maxThreads, n);
   numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));

   // Copies the elements to the device
   typename Input1Iterator::device_pointer_type_cu in1_mem_p = input1Begin.getParent().updateDevice_CU(input1Begin.getAddress(), n1, deviceID, true, false);
   typename Input2Iterator::device_pointer_type_cu in2_mem_p = input2Begin.getParent().updateDevice_CU(input2Begin.getAddress(), n, deviceID, true, false);
   typename OutputIterator::device_pointer_type_cu out_mem_p = outputBegin.getParent().updateDevice_CU(outputBegin.getAddress(), n, deviceID, false, true);

   // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
   MapArrayKernel_CU<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapArrayFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#else
   MapArrayKernel_CU<<<numBlocks,numThreads>>>(*m_mapArrayFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#endif

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();

#ifdef TUNER_MODE
   cudaDeviceSynchronize();
#endif
}


/*!
 *  Performs the MapArray on the two vectors with \em CUDA as backend. Seperate output vector.
 *  First Vector can be accessed entirely for each element in second Vector.
 *  The function is a wrapper for
 *  CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU).
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CU(Vector<T>& input1, Vector<T>& input2, Vector<T>& output, int useNumGPU)
{
   if(input2.size() != output.size())
   {
      output.clear();
      output.resize(input2.size());
   }

   CU(input1.begin(), input1.end(), input2.begin(), input2.end(), output.begin(), useNumGPU);
}






/*!
 *  Performs MapArray on the one vector and one sparse matrix block-wise with \em CUDA as backend. Seperate output vector is used.
 *  The Vector can be accessed entirely for "a block of elements" in the SparseMatrix. The block-length is specified in the user-function.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CU(Vector<T>& input1, SparseMatrix<T>& input2, Vector<T>& output, int useNumGPU)
{
   cudaGetLastError();

   size_t nrows = input2.total_rows();
   size_t ncols = input2.total_cols();

   size_t nnz = input2.total_nnz();

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

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      unsigned int deviceID = 0;

      CHECK_CUDA_ERROR(cudaSetDevice(deviceID));

      // Setup parameters
      size_t n1 = input1.size();
      BackEndParams bp = m_execPlan->find_(outSize);
      size_t maxBlocks = bp.maxBlocks; //Chosen somewhat arbitrarly
      size_t maxThreads = bp.maxThreads;
      size_t numBlocks;
      size_t numThreads;

      numThreads = std::min(maxThreads, outSize); // one thread per output element
      numBlocks = std::max((size_t)1, std::min( (outSize/numThreads + (outSize%numThreads == 0 ? 0:1)), maxBlocks));

      // Copies the elements to the device
      typename Vector<T>::device_pointer_type_cu in1_mem_p = input1.updateDevice_CU(input1.getAddress(), n1, deviceID, true, false);
      typename SparseMatrix<T>::device_pointer_type_cu in2_mem_p = input2.updateDevice_CU(input2.get_values(), nnz, deviceID, true);

      typename Vector<T>::device_pointer_type_cu out_mem_p = output.updateDevice_CU(output.getAddress(), outSize, deviceID, false, true);

      typename SparseMatrix<T>::device_pointer_index_type_cu in2_row_offsets_mem_p = input2.updateDevice_Index_CU(input2.get_row_pointers(), nrows+1, deviceID, true);
      typename SparseMatrix<T>::device_pointer_index_type_cu in2_col_indices_mem_p = input2.updateDevice_Index_CU(input2.get_col_indices(), nnz, deviceID, true);

      // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
      MapArrayKernel_CU_Sparse_Matrix_Blockwise<<<numBlocks,numThreads, 0,  (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapArrayFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), in2_row_offsets_mem_p->getDeviceDataPointer(), in2_col_indices_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), outSize, 0);
#else
      MapArrayKernel_CU_Sparse_Matrix_Blockwise<<<numBlocks,numThreads>>>(*m_mapArrayFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), in2_row_offsets_mem_p->getDeviceDataPointer(), in2_col_indices_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), outSize, 0);
#endif

      cutilCheckMsg("The MapArray kernel error.");

      // Make sure the data is marked as changed by the device
      out_mem_p->changeDeviceData();

#ifdef TUNER_MODE
      cudaDeviceSynchronize();
#endif
   }
   else
   {
      size_t n1 = input1.size();

      size_t numElemPerSlice = outSize / numDevices;

      size_t restElem = outSize % numDevices;

      typename Vector<T>::device_pointer_type_cu in1_mem_p[MAX_GPU_DEVICES];
      typename SparseMatrix<T>::device_pointer_type_cu in2_mem_p[MAX_GPU_DEVICES];
      typename Vector<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      typename SparseMatrix<T>::device_pointer_index_type_cu in2_row_offsets_mem_p[MAX_GPU_DEVICES];
      typename SparseMatrix<T>::device_pointer_index_type_cu in2_col_indices_mem_p[MAX_GPU_DEVICES];

      size_t offset = 0;
      size_t end = 0;

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+restElem;
         else
            numElem = numElemPerSlice;

         end = input2.get_rowOffsetFromStart(numElem+i*numElemPerSlice);

         in1_mem_p[i] = input1.updateDevice_CU(input1.getAddress(), n1, i, false, false);
         in2_mem_p[i] = input2.updateDevice_CU((input2.get_values()+offset), end-offset, i, false);

         in2_row_offsets_mem_p[i] = input2.updateDevice_Index_CU(((input2.get_row_pointers())+i*numElemPerSlice), numElem+1, i, false);
         in2_col_indices_mem_p[i] = input2.updateDevice_Index_CU((input2.get_col_indices()+offset), end-offset, i, false);

         out_mem_p[i] = output.updateDevice_CU(((output.getAddress())+i*numElemPerSlice), numElem, i, false, false);

         offset = end;
      }

      offset = end = 0;

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+restElem;
         else
            numElem = numElemPerSlice;

         end = input2.get_rowOffsetFromStart(numElem+i*numElemPerSlice);

         // Setup parameters
         BackEndParams bp=m_execPlan->find_(numElem);
         size_t maxBlocks = bp.maxBlocks; //Chosen somewhat arbitrarly
         size_t maxThreads = bp.maxThreads;
         size_t numBlocks;
         size_t numThreads;

         numThreads = std::min(maxThreads, numElem); // one thread per output element
         numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

         // Copies the elements to the device
         in1_mem_p[i] = input1.updateDevice_CU(input1.getAddress(), n1, i, true, false);
         in2_mem_p[i] = input2.updateDevice_CU((input2.get_values()+offset), end-offset, i, true);
         out_mem_p[i] = output.updateDevice_CU(((output.getAddress())+i*numElemPerSlice), numElem, i, false, true, true);

         in2_row_offsets_mem_p[i] = input2.updateDevice_Index_CU(((input2.get_row_pointers())+i*numElemPerSlice), numElem+1, i, true);
         in2_col_indices_mem_p[i] = input2.updateDevice_Index_CU((input2.get_col_indices()+offset), end-offset, i, true);

         // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
         MapArrayKernel_CU_Sparse_Matrix_Blockwise<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapArrayFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), in2_row_offsets_mem_p[i]->getDeviceDataPointer(), in2_col_indices_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem, offset);
#else
         MapArrayKernel_CU_Sparse_Matrix_Blockwise<<<numBlocks,numThreads>>>(*m_mapArrayFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), in2_row_offsets_mem_p[i]->getDeviceDataPointer(), in2_col_indices_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem, offset);
#endif

         cutilCheckMsg("The MapArray kernel error.");

         // Make sure the data is marked as changed by the device
         out_mem_p[i]->changeDeviceData();

         offset = end;
      }
      
      CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
      
      output.setValidFlag(false); // set parent copy to invalid...
   }
}



/*!
 *  Performs MapArray on the one vector and one matrix block-wise with \em CUDA as backend. Seperate output vector is used.
 *  The Vector can be accessed entirely for "a block of elements" in the Matrix. The block-length is specified in the user-function.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CU(Vector<T>& input1, Matrix<T>& input2, Vector<T>& output, int useNumGPU)
{
   cudaGetLastError();

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

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      unsigned int deviceID = 0;

      CHECK_CUDA_ERROR(cudaSetDevice(deviceID));

      // Setup parameters
      size_t n1 = input1.size();
      BackEndParams bp = m_execPlan->find_(outSize);
      size_t maxBlocks = bp.maxBlocks; //Chosen somewhat arbitrarly
      size_t maxThreads = bp.maxThreads;
      size_t numBlocks;
      size_t numThreads;

      numThreads = std::min(maxThreads, outSize); // one thread per output element
      numBlocks = std::max((size_t)1, std::min( (outSize/numThreads + (outSize%numThreads == 0 ? 0:1)), maxBlocks));

      // Copies the elements to the device
      typename Vector<T>::device_pointer_type_cu in1_mem_p = input1.updateDevice_CU(input1.getAddress(), n1, deviceID, true, false);
      typename Matrix<T>::device_pointer_type_cu in2_mem_p = input2.updateDevice_CU(input2.getAddress(), nsize, deviceID, true, false);
      typename Vector<T>::device_pointer_type_cu out_mem_p = output.updateDevice_CU(output.getAddress(), outSize, deviceID, false, true);

      // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
      MapArrayKernel_CU_Matrix_Blockwise<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapArrayFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), outSize, p2BlockSize);
#else
      MapArrayKernel_CU_Matrix_Blockwise<<<numBlocks,numThreads>>>(*m_mapArrayFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), outSize, p2BlockSize);
#endif

      cutilCheckMsg("The MapArray kernel error.");

      // Make sure the data is marked as changed by the device
      out_mem_p->changeDeviceData();

#ifdef TUNER_MODE
      cudaDeviceSynchronize();
#endif
   }
   else
   {
      size_t n1 = input1.size();

      size_t numElemPerSlice = outSize / numDevices;

      size_t restElem = outSize % numDevices;

      typename Vector<T>::device_pointer_type_cu in1_mem_p[MAX_GPU_DEVICES];
      typename Matrix<T>::device_pointer_type_cu in2_mem_p[MAX_GPU_DEVICES];
      typename Vector<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+restElem;
         else
            numElem = numElemPerSlice;

         in1_mem_p[i] = input1.updateDevice_CU(input1.getAddress(), n1, i, false, false);
         in2_mem_p[i] = input2.updateDevice_CU(((input2.getAddress())+i*numElemPerSlice*p2BlockSize), numElem*p2BlockSize, i, false, false);
         out_mem_p[i] = output.updateDevice_CU(((output.getAddress())+i*numElemPerSlice), numElem, i, false, false);
      }

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+restElem;
         else
            numElem = numElemPerSlice;

         // Setup parameters
         BackEndParams bp=m_execPlan->find_(numElem);
         size_t maxBlocks = bp.maxBlocks; //Chosen somewhat arbitrarly
         size_t maxThreads = bp.maxThreads;
         size_t numBlocks;
         size_t numThreads;

         numThreads = std::min(maxThreads, numElem); // one thread per output element
         numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

         // Copies the elements to the device
         in1_mem_p[i] = input1.updateDevice_CU(input1.getAddress(), n1, i, true, false);
         in2_mem_p[i] = input2.updateDevice_CU(((input2.getAddress())+i*numElemPerSlice*p2BlockSize), numElem*p2BlockSize, i, true, false);
         out_mem_p[i] = output.updateDevice_CU(((output.getAddress())+i*numElemPerSlice), numElem, i, false, true, true);

         // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
         MapArrayKernel_CU_Matrix_Blockwise<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapArrayFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem, p2BlockSize);
#else
         MapArrayKernel_CU_Matrix_Blockwise<<<numBlocks,numThreads>>>(*m_mapArrayFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem, p2BlockSize);
#endif

         cutilCheckMsg("The MapArray kernel error.");

         // Make sure the data is marked as changed by the device
         out_mem_p[i]->changeDeviceData();
      }

      CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
      
      output.setValidFlag(false); // set parent copy to invalid...
   }
}


/*!
 *  Performs the MapArray on the one vector and one matrix with \em CUDA as backend. Seperate output matrix.
 *  The vector can be accessed entirely for each element in the matrix.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = Implementation decides.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::CU(Vector<T>& input1, Matrix<T>& input2, Matrix<T>& output, int useNumGPU)
{
   cudaGetLastError();

   if(input2.size() != output.size())
   {
      output.clear();
      output.resize(input2.total_rows(), input2.total_cols());
   }

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      unsigned int deviceID = 0;

      CHECK_CUDA_ERROR(cudaSetDevice(deviceID));

      // Setup parameters
      size_t nrows = input2.total_rows();
      size_t ncols = input2.total_cols();
      size_t n = nrows * ncols;
      size_t n1 = input1.size();

      dim3 numThreads, numBlocks;

      numThreads.x =  (ncols>32)? 32: ncols;                            // each thread does multiple Xs
      numThreads.y =  (nrows>16)? 16: nrows;
      numThreads.z = 1;
      numBlocks.x = (ncols+(numThreads.x-1)) / numThreads.x;
      numBlocks.y = (nrows+(numThreads.y-1))  / numThreads.y;
      numBlocks.z = 1;

      // Copies the elements to the device
      typename Vector<T>::device_pointer_type_cu in1_mem_p = input1.updateDevice_CU(input1.getAddress(), n1, deviceID, true, false);
      typename Matrix<T>::device_pointer_type_cu in2_mem_p = input2.updateDevice_CU(input2.getAddress(), n, deviceID, true, false);
      typename Matrix<T>::device_pointer_type_cu out_mem_p = output.updateDevice_CU(output.getAddress(), n, deviceID, false, true);

      // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
      MapArrayKernel_CU_Matrix<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapArrayFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n, ncols, nrows, 0);
#else
      MapArrayKernel_CU_Matrix<<<numBlocks,numThreads>>>(*m_mapArrayFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n, ncols, nrows, 0);
#endif

      cutilCheckMsg("The MapArray kernel error.");

      // Make sure the data is marked as changed by the device
      out_mem_p->changeDeviceData();

#ifdef TUNER_MODE
      cudaDeviceSynchronize();
#endif
   }
   else
   {
      size_t nrows = input2.total_rows();
      size_t ncols = input2.total_cols();
      size_t n1 = input1.size();

      size_t numRowsPerSlice = nrows / numDevices;
      size_t numElemPerSlice = numRowsPerSlice * ncols;

      size_t restRows = nrows % numDevices;

      typename Vector<T>::device_pointer_type_cu in1_mem_p[MAX_GPU_DEVICES];
      typename Matrix<T>::device_pointer_type_cu in2_mem_p[MAX_GPU_DEVICES];
      typename Matrix<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem, numRows;
         if(i == numDevices-1)
            numRows = numRowsPerSlice+restRows;
         else
            numRows = numRowsPerSlice;

         numElem = numRows * ncols;

         in1_mem_p[i] = input1.updateDevice_CU(input1.getAddress(), n1, i, false, false);
         in2_mem_p[i] = input2.updateDevice_CU(((input2.getAddress())+i*numElemPerSlice), numElem, i, false, false);
         out_mem_p[i] = output.updateDevice_CU(((output.getAddress())+i*numElemPerSlice), numElem, i, false, false);
      }


      size_t yoffset = 0;

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem, numRows;
         if(i == numDevices-1)
            numRows = numRowsPerSlice+restRows;
         else
            numRows = numRowsPerSlice;

         numElem = numRows * ncols;

         // Setup parameters
         dim3 numThreads, numBlocks;

         numThreads.x =  (ncols>32)? 32: ncols;                            // each thread does multiple Xs
         numThreads.y =  (numRows>16)? 16: numRows;
         numThreads.z = 1;
         numBlocks.x = (ncols+(numThreads.x-1)) / numThreads.x;
         numBlocks.y = (numRows+(numThreads.y-1))  / numThreads.y;
         numBlocks.z = 1;

         // Copies the elements to the device
         in1_mem_p[i] = input1.updateDevice_CU(input1.getAddress(), n1, i, true, false);
         in2_mem_p[i] = input2.updateDevice_CU(((input2.getAddress())+i*numElemPerSlice), numElem, i, true, false);
         out_mem_p[i] = output.updateDevice_CU(((output.getAddress())+i*numElemPerSlice), numElem, i, false, true, true);

         // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
         MapArrayKernel_CU_Matrix<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapArrayFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem, ncols, numRows, yoffset);
#else
         MapArrayKernel_CU_Matrix<<<numBlocks,numThreads>>>(*m_mapArrayFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem, ncols, numRows, yoffset);
#endif

         cutilCheckMsg("The MapArray kernel error.");

         yoffset += numRows;

         // Make sure the data is marked as changed by the device
         out_mem_p[i]->changeDeviceData();
      }

      CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
      
      output.setValidFlag(false); // set parent copy to invalid...
   }
}


/*!
 *  Performs the MapArray on the two element ranges with \em  CUDA as backend. Seperate output range.
 *  Decides whether to use one device and mapArraySingleThread_CU or multiple devices.
 *  In the case of several devices the input ranges is divided evenly
 *  among the threads created. First range can be accessed entirely for each element in second range.
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
void MapArray<MapArrayFunc>::CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAPARRAY CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapArraySingleThread_CU(input1Begin, input1End, input2Begin, input2End, outputBegin, (m_environment->bestCUDADevID));
   }
   else
   {
      size_t n = input2End - input2Begin;
      size_t n1 = input1End - input1Begin;

      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename Input1Iterator::device_pointer_type_cu in1_mem_p[MAX_GPU_DEVICES];
      typename Input2Iterator::device_pointer_type_cu in2_mem_p[MAX_GPU_DEVICES];
      typename OutputIterator::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in1_mem_p[i] = input1Begin.getParent().updateDevice_CU(input1Begin.getAddress(), n1, i, false, false);
         in2_mem_p[i] = input2Begin.getParent().updateDevice_CU((input2Begin+i*numElemPerSlice).getAddress(), numElem, i, false, false);
         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem, i, false, false);
      }

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         CHECK_CUDA_ERROR(cudaSetDevice(i));

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         // Setup parameters
         BackEndParams bp=m_execPlan->find_(numElem);
         size_t maxBlocks = bp.maxBlocks;; //Chosen somewhat arbitrarly
         size_t maxThreads = bp.maxThreads;
         size_t numBlocks;
         size_t numThreads;

         numThreads = std::min(maxThreads, n);
         numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));

         // Copies the elements to the device
         in1_mem_p[i] = input1Begin.getParent().updateDevice_CU(input1Begin.getAddress(), n1, i, true, false);
         in2_mem_p[i] = input2Begin.getParent().updateDevice_CU((input2Begin+i*numElemPerSlice).getAddress(), numElem, i, true, false);
         out_mem_p[i] = outputBegin.getParent().updateDevice_CU((outputBegin+i*numElemPerSlice).getAddress(), numElem, i, false, true, true);

         // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
         MapArrayKernel_CU<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapArrayFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#else
         MapArrayKernel_CU<<<numBlocks,numThreads>>>(*m_mapArrayFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#endif

         // Make sure the data is marked as changed by the device
         out_mem_p[i]->changeDeviceData();
      }

      CHECK_CUDA_ERROR(cudaSetDevice(m_environment->bestCUDADevID));
   }
   outputBegin.getParent().setValidFlag(false); // set parent copy to invalid...
}


}

#endif

