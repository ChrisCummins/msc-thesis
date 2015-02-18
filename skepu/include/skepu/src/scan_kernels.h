/*! \file scan_kernels.h
 *  \brief Contains the OpenCL and CUDA kernels for the Scan skeleton.
 */

#ifndef SCAN_KERNELS_H
#define SCAN_KERNELS_H

#ifdef SKEPU_OPENCL

#include <string>

namespace skepu
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup ScanKernels Scan Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Scan skeleton.
 * \{
 */

/*!
 *
 *  OpenCL Scan kernel. The algorithm is a variation of the naive parallel scan algorithm.
 *  Supports exclusive and inclusive scans as well as user defined operators. Also handles
 *  vectors of any size.
 */
static std::string ScanKernel_CL(
   "__kernel void ScanKernel_KERNELNAME(__global TYPE* input, __global TYPE* output, __global TYPE* blockSums, size_t n, size_t numElements, __local TYPE* sdata)\n"
   "{\n"
   "    size_t threadIdx = get_local_id(0);\n"
   "    size_t blockDim = get_local_size(0);\n"
   "    size_t blockIdx = get_group_id(0);\n"
   "    size_t gridDim = get_num_groups(0);\n"
   "    size_t thid = threadIdx;\n"
   "    unsigned int pout = 0;\n"
   "    unsigned int pin = 1;\n"
   "    size_t mem = get_global_id(0);\n"
   "    size_t blockNr = blockIdx;\n"
   "    size_t gridSize = blockDim*gridDim;\n"
   "    size_t numBlocks = numElements/(blockDim) + (numElements%(blockDim) == 0 ? 0:1);\n"
   "    size_t offset;\n"
   "    while(blockNr < numBlocks)\n"
   "    {\n"
   "        sdata[pout*n+thid] = (mem < numElements) ? input[mem] : 0;\n"
   "        barrier(CLK_LOCAL_MEM_FENCE);\n"
   "        for(offset = 1; offset < n; offset *=2)\n"
   "        {\n"
   "            pout = 1-pout;\n"
   "            pin = 1-pout;\n"
   "            if(thid >= offset)\n"
   "                sdata[pout*n+thid] = FUNCTIONNAME(sdata[pin*n+thid], sdata[pin*n+thid-offset], (TYPE)0);\n"
   "            else\n"
   "                sdata[pout*n+thid] = sdata[pin*n+thid];\n"
   "            barrier(CLK_LOCAL_MEM_FENCE);\n"
   "        }\n"
   "        if(thid == blockDim - 1)\n"
   "            blockSums[blockNr] = sdata[pout*n+blockDim-1];\n"
   "        if(mem < numElements)\n"
   "            output[mem] = sdata[pout*n+thid];\n"
   "        mem += gridSize;\n"
   "        blockNr += gridDim;\n"
   "        barrier(CLK_LOCAL_MEM_FENCE);\n"
   "    }\n"
   "}\n"
);

/*!
 *
 *  Helper kernel to the OpenCL Scan kernel. Applies the offsets for each block that were saved in main
 *  Scan kernel.
 */
static std::string ScanUpdate_CL(
   "__kernel void ScanUpdate_KERNELNAME(__global TYPE* data, __global TYPE* sums, int isInclusive, TYPE init, size_t n, __global TYPE* ret, __local TYPE* sdata)\n"
   "{\n"
   "    __local TYPE offset;\n"
   "    __local TYPE inc_offset;\n"
   "    size_t threadIdx = get_local_id(0);\n"
   "    size_t blockDim = get_local_size(0);\n"
   "    size_t blockIdx = get_group_id(0);\n"
   "    size_t gridDim = get_num_groups(0);\n"
   "    size_t thid = threadIdx;\n"
   "    size_t blockNr = blockIdx;\n"
   "    size_t gridSize = blockDim*gridDim;\n"
   "    size_t mem = get_global_id(0);\n"
   "    size_t numBlocks = n/(blockDim) + (n%(blockDim) == 0 ? 0:1);\n"
   "    while(blockNr < numBlocks)\n"
   "    {\n"
   "        if(thid == 0)\n"
   "        {\n"
   "            if(isInclusive == 0)\n"
   "            {\n"
   "                offset = init;\n"
   "                if(blockNr > 0)\n"
   "                {\n"
   "                    offset = FUNCTIONNAME(offset, sums[blockNr-1], (TYPE)0);\n"
   "                    inc_offset = sums[blockNr-1];\n"
   "                }\n"
   "            }\n"
   "            else\n"
   "            {\n"
   "                if(blockNr > 0)\n"
   "                    offset = sums[blockNr-1];\n"
   "            }\n"
   "        }\n"
   "        barrier(CLK_LOCAL_MEM_FENCE);\n"
   "        if(isInclusive == 1)\n"
   "        {\n"
   "            if(blockNr > 0)\n"
   "                sdata[thid] = (mem < n) ? FUNCTIONNAME(offset, data[mem], (TYPE)0) : 0;\n"
   "            else\n"
   "                sdata[thid] = (mem < n) ? data[mem] : 0;\n"
   "            if(mem == n-1)\n"
   "                *ret = sdata[thid];\n"
   "        }\n"
   "        else\n"
   "        {\n"
   "            if(mem == n-1)\n"
   "                *ret = FUNCTIONNAME(inc_offset, data[mem], (TYPE)0);\n"
   "            if(thid == 0)\n"
   "                sdata[thid] = offset;\n"
   "            else\n"
   "                sdata[thid] = (mem-1 < n) ? FUNCTIONNAME(offset, data[mem-1], (TYPE)0) : 0;\n"
   "        }\n"
   "        barrier(CLK_LOCAL_MEM_FENCE);\n"
   "        if(mem < n)\n"
   "            data[mem] = sdata[thid];\n"
   "        mem += gridSize;\n"
   "        blockNr += gridDim;\n"
   "        barrier(CLK_LOCAL_MEM_FENCE);\n"
   "    }\n"
   "}\n"
);

/*!
 *
 *  Helper kernel to the OpenCL multi-GPU Scan. Applies the device sums to each devices work batch.
 */
static std::string ScanAdd_CL(
   "__kernel void ScanAdd_KERNELNAME(__global TYPE* data, TYPE sum, size_t n)\n"
   "{\n"
   "    size_t i = get_global_id(0);\n"
   "    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"
   "    while(i < n)\n"
   "    {\n"
   "        data[i] = FUNCTIONNAME(data[i], sum, (TYPE)0);\n"
   "        i += gridSize;\n"
   "    }\n"
   "}\n"
);

/*!
 *  \}
 */

}

#endif

#ifdef SKEPU_CUDA

namespace skepu
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup ScanKernels Scan Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Scan skeleton.
 * \{
 */

/*!
 *
 *  CUDA Scan kernel. The algorithm is a variation of the naive parallel scan algorithm.
 *  Supports exclusive and inclusive scans as well as user defined operators. Also handles
 *  vectors of any size.
 */
template<typename T, typename BinaryFunc>
__global__ void ScanKernel_CU(BinaryFunc scanFunc, T* input, T* output, T* blockSums, size_t n, size_t numElements)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata);

   size_t thid = threadIdx.x;
   unsigned int pout = 0;
   unsigned int pin = 1;
   size_t mem = blockIdx.x * blockDim.x + threadIdx.x;
   size_t blockNr = blockIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;
   size_t numBlocks = numElements/(blockDim.x) + (numElements%(blockDim.x) == 0 ? 0:1);
   size_t offset;

   while(blockNr < numBlocks)
   {
      sdata[pout*n+thid] = (mem < numElements) ? input[mem] : 0;

      __syncthreads();

      for(offset = 1; offset < n; offset *=2)
      {
         pout = 1-pout;
         pin = 1-pout;
         if(thid >= offset)
            sdata[pout*n+thid] = scanFunc.CU(sdata[pin*n+thid], sdata[pin*n+thid-offset]);
         else
            sdata[pout*n+thid] = sdata[pin*n+thid];

         __syncthreads();
      }
      if(thid == blockDim.x - 1)
         blockSums[blockNr] = sdata[pout*n+blockDim.x-1];

      if(mem < numElements)
         output[mem] = sdata[pout*n+thid];

      mem += gridSize;
      blockNr += gridDim.x;

      __syncthreads();
   }
}

/*!
 *
 *  Helper kernel to the CUDA Scan kernel. Applies the offsets for each block that were saved in main
 *  Scan kernel.
 */
template <typename T, typename BinaryFunc>
__global__ void ScanUpdate_CU(BinaryFunc scanFunc, T *data, T *sums, int isInclusive, T init, size_t n, T *ret)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata);

   __shared__ T offset;
   __shared__ T inc_offset;

   size_t thid = threadIdx.x;
   size_t blockNr = blockIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;
   size_t mem = blockIdx.x * blockDim.x + threadIdx.x;
   size_t numBlocks = n/(blockDim.x) + (n%(blockDim.x) == 0 ? 0:1);

   while(blockNr < numBlocks)
   {
      if(thid == 0)
      {
         if(isInclusive == 0)
         {
            offset = init;
            if(blockNr > 0)
            {
               offset = scanFunc.CU(offset, sums[blockNr-1]);
               inc_offset = sums[blockNr-1];
            }
         }
         else
         {
            if(blockNr > 0)
               offset = sums[blockNr-1];
         }
      }

      __syncthreads();

      if(isInclusive == 1)
      {
         if(blockNr > 0)
            sdata[thid] = (mem < n) ? scanFunc.CU(offset, data[mem]) : 0;
         else
            sdata[thid] = (mem < n) ? data[mem] : 0;
         if(mem == n-1)
            *ret = sdata[thid];
      }
      else
      {
         if(mem == n-1)
            *ret = scanFunc.CU(inc_offset, data[mem]);
         if(thid == 0)
            sdata[thid] = offset;
         else
            sdata[thid] = (mem-1 < n) ? scanFunc.CU(offset, data[mem-1]) : 0;
      }

      __syncthreads();

      if(mem < n)
         data[mem] = sdata[thid];

      mem += gridSize;
      blockNr += gridDim.x;

      __syncthreads();
   }
}

/*!
 *
 *  Helper kernel to the CUDA multi-GPU Scan. Applies the device sums to each devices work batch.
 */
template <typename T, typename BinaryFunc>
__global__ void ScanAdd_CU(BinaryFunc scanFunc, T *data, T sum, size_t n)
{
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   while(i < n)
   {
      data[i] = scanFunc.CU(data[i], sum);
      i += gridSize;
   }
}

/*!
 *  \}
 */

}

#endif

#endif

