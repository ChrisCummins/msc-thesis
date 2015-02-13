/*! \file mapreduce_kernels.h
 *  \brief Contains the OpenCL and CUDA kernels for the MapReduce skeleton.
 */

#ifndef MAPREDUCE_KERNELS_H
#define MAPREDUCE_KERNELS_H

#ifdef SKEPU_OPENCL

#include <string>

namespace skepu
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup MapReduceKernels MapReduce Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the MapReduce skeleton.
 * \{
 */

/*!
 *
 *  OpenCL MapReduce kernel, using the same pattern as reduce6 in the CUDA SDK. See whitepaper from NVIDIA on optimizing
 *  reduction for the GPU. For unary map user functions.
 */
static std::string UnaryMapReduceKernel_CL
(
   "__kernel void UnaryMapReduceKernel_KERNELNAME(__global TYPE* input, __global TYPE* output, size_t n, __local TYPE* sdata, CONST_TYPE const1)\n"
   "{\n"
   "    size_t blockSize = get_local_size(0);\n"
   "    size_t tid = get_local_id(0);\n"
   "    size_t i = get_group_id(0)*blockSize + get_local_id(0);\n"
   "    size_t gridSize = blockSize*get_num_groups(0);\n"
   "    TYPE result = 0;\n"
   "    if(i < n)\n"
   "    {\n"
   "        result = FUNCTIONNAME_MAP(input[i], const1);\n"
   "        i += gridSize;\n"
   "    }\n"
   "    while(i < n)\n"
   "    {\n"
   "        TYPE tempMap;\n"
   "        tempMap = FUNCTIONNAME_MAP(input[i], const1);\n"
   "        result = FUNCTIONNAME_REDUCE(result, tempMap, (TYPE)0);\n"
   "        i += gridSize;\n"
   "    }\n"
   "    sdata[tid] = result;\n"
   "    barrier(CLK_LOCAL_MEM_FENCE);\n"
   "    if(blockSize >= 512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid + 256], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >= 256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid + 128], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >= 128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +  64], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +  32], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +  16], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   8], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   4], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   2], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   1], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(tid == 0)\n"
   "    {\n"
   "        output[get_group_id(0)] = sdata[tid];\n"
   "    }\n"
   "}\n"
);

/*!
 *
 *  OpenCL MapReduce kernel, using the same pattern as reduce6 in the CUDA SDK. See whitepaper from NVIDIA on optimizing
 *  reduction for the GPU. For binary map user functions.
 */
static std::string BinaryMapReduceKernel_CL
(
   "__kernel void BinaryMapReduceKernel_KERNELNAME(__global TYPE* input1, __global TYPE* input2, __global TYPE* output, size_t n, __local TYPE* sdata, CONST_TYPE const1)\n"
   "{\n"
   "    size_t blockSize = get_local_size(0);\n"
   "    size_t tid = get_local_id(0);\n"
   "    size_t i = get_group_id(0)*blockSize + get_local_id(0);\n"
   "    size_t gridSize = blockSize*get_num_groups(0);\n"
   "    TYPE result = 0;\n"
   "    if(i < n)\n"
   "    {\n"
   "        result = FUNCTIONNAME_MAP(input1[i], input2[i], const1);\n"
   "        i += gridSize;\n"
   "    }\n"
   "    while(i < n)\n"
   "    {\n"
   "        TYPE tempMap;\n"
   "        tempMap = FUNCTIONNAME_MAP(input1[i], input2[i], const1);\n"
   "        result = FUNCTIONNAME_REDUCE(result, tempMap, (TYPE)0);\n"
   "        i += gridSize;\n"
   "    }\n"
   "    sdata[tid] = result;\n"
   "    barrier(CLK_LOCAL_MEM_FENCE);\n"
   "    if(blockSize >= 512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid + 256], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >= 256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid + 128], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >= 128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +  64], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +  32], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +  16], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   8], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   4], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   2], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   1], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(tid == 0)\n"
   "    {\n"
   "        output[get_group_id(0)] = sdata[tid];\n"
   "    }\n"
   "}\n"
);

/*!
 *
 *  OpenCL MapReduce kernel, using the same pattern as reduce6 in the CUDA SDK. See whitepaper from NVIDIA on optimizing
 *  reduction for the GPU. For trinary map user functions.
 */
static std::string TrinaryMapReduceKernel_CL
(
   "__kernel void TrinaryMapReduceKernel_KERNELNAME(__global TYPE* input1, __global TYPE* input2, __global TYPE* input3, __global TYPE* output, size_t n, __local TYPE* sdata, CONST_TYPE const1)\n"
   "{\n"
   "    size_t blockSize = get_local_size(0);\n"
   "    size_t tid = get_local_id(0);\n"
   "    size_t i = get_group_id(0)*blockSize + get_local_id(0);\n"
   "    size_t gridSize = blockSize*get_num_groups(0);\n"
   "    TYPE result = 0;\n"
   "    if(i < n)\n"
   "    {\n"
   "        result = FUNCTIONNAME_MAP(input1[i], input2[i], input3[i], const1);\n"
   "        i += gridSize;\n"
   "    }\n"
   "    while(i < n)\n"
   "    {\n"
   "        TYPE tempMap;\n"
   "        tempMap = FUNCTIONNAME_MAP(input1[i], input2[i], input3[i], const1);\n"
   "        result = FUNCTIONNAME_REDUCE(result, tempMap, (TYPE)0);\n"
   "        i += gridSize;\n"
   "    }\n"
   "    sdata[tid] = result;\n"
   "    barrier(CLK_LOCAL_MEM_FENCE);\n"
   "    if(blockSize >= 512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid + 256], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >= 256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid + 128], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >= 128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +  64], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +  32], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +  16], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   8], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   4], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   2], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = FUNCTIONNAME_REDUCE(sdata[tid], sdata[tid +   1], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(tid == 0)\n"
   "    {\n"
   "        output[get_group_id(0)] = sdata[tid];\n"
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
 *  \defgroup MapReduceKernels MapReduce Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Reduce skeleton.
 * \{
 */

/*!
 *
 *  CUDA MapReduce kernel, using the same pattern as reduce6 in the CUDA SDK. See whitepaper from NVIDIA on optimizing
 *  reduction for the GPU. For unary map user functions.
 */
template <typename T, typename UnaryFunc, typename BinaryFunc>
__global__ void MapReduceKernel1_CU(UnaryFunc mapFunc, BinaryFunc reduceFunc, T* input, T* output, size_t n)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata);

   size_t blockSize = blockDim.x;
   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockSize + tid;
   size_t gridSize = blockSize*gridDim.x;
   T result = 0;

   if(i < n)
   {
      result = mapFunc.CU(input[i]);
      i += gridSize;
   }

   while(i < n)
   {
      T tempMap;
      tempMap = mapFunc.CU(input[i]);
      result = reduceFunc.CU(result, tempMap);
      i += gridSize;
   }

   sdata[tid] = result;

   __syncthreads();

   if(blockSize >= 512)
   {
      if (tid < 256 && tid + 256 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid + 256]);
      }
      __syncthreads();
   }
   if(blockSize >= 256)
   {
      if (tid < 128 && tid + 128 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid + 128]);
      }
      __syncthreads();
   }
   if(blockSize >= 128)
   {
      if (tid <  64 && tid +  64 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +  64]);
      }
      __syncthreads();
   }
   if(blockSize >=  64)
   {
      if (tid <  32 && tid +  32 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +  32]);
      }
      __syncthreads();
   }
   if(blockSize >=  32)
   {
      if (tid <  16 && tid +  16 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +  16]);
      }
      __syncthreads();
   }
   if(blockSize >=  16)
   {
      if (tid <   8 && tid +   8 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   8]);
      }
      __syncthreads();
   }
   if(blockSize >=   8)
   {
      if (tid <   4 && tid +   4 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   4]);
      }
      __syncthreads();
   }
   if(blockSize >=   4)
   {
      if (tid <   2 && tid +   2 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   2]);
      }
      __syncthreads();
   }
   if(blockSize >=   2)
   {
      if (tid <   1 && tid +   1 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   1]);
      }
      __syncthreads();
   }

   if(tid == 0)
   {
      output[blockIdx.x] = sdata[tid];
   }
}

/*!
 *
 *  CUDA MapReduce kernel, using the same pattern as reduce6 in the CUDA SDK. See whitepaper from NVIDIA on optimizing
 *  reduction for the GPU. For binary map user functions.
 */
template <typename T, typename BinaryFunc1, typename BinaryFunc2>
__global__ void MapReduceKernel2_CU(BinaryFunc1 mapFunc, BinaryFunc2 reduceFunc, T* input1, T* input2, T* output, size_t n)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata);

   size_t blockSize = blockDim.x;
   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockSize + tid;
   size_t gridSize = blockSize*gridDim.x;
   T result = 0;

   if(i < n)
   {
      result = mapFunc.CU(input1[i], input2[i]);
      i += gridSize;
   }

   while(i < n)
   {
      T tempMap;
      tempMap = mapFunc.CU(input1[i], input2[i]);
      result = reduceFunc.CU(result, tempMap);
      i += gridSize;
   }

   sdata[tid] = result;

   __syncthreads();

   if(blockSize >= 512)
   {
      if (tid < 256 && tid + 256 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid + 256]);
      }
      __syncthreads();
   }
   if(blockSize >= 256)
   {
      if (tid < 128 && tid + 128 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid + 128]);
      }
      __syncthreads();
   }
   if(blockSize >= 128)
   {
      if (tid <  64 && tid +  64 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +  64]);
      }
      __syncthreads();
   }
   if(blockSize >=  64)
   {
      if (tid <  32 && tid +  32 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +  32]);
      }
      __syncthreads();
   }
   if(blockSize >=  32)
   {
      if (tid <  16 && tid +  16 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +  16]);
      }
      __syncthreads();
   }
   if(blockSize >=  16)
   {
      if (tid <   8 && tid +   8 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   8]);
      }
      __syncthreads();
   }
   if(blockSize >=   8)
   {
      if (tid <   4 && tid +   4 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   4]);
      }
      __syncthreads();
   }
   if(blockSize >=   4)
   {
      if (tid <   2 && tid +   2 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   2]);
      }
      __syncthreads();
   }
   if(blockSize >=   2)
   {
      if (tid <   1 && tid +   1 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   1]);
      }
      __syncthreads();
   }

   if(tid == 0)
   {
      output[blockIdx.x] = sdata[tid];
   }
}

/*!
 *
 *  CUDA MapReduce kernel, using the same pattern as reduce6 in the CUDA SDK. See whitepaper from NVIDIA on optimizing
 *  reduction for the GPU. For trinary map user functions.
 */
template <typename T, typename TrinaryFunc, typename BinaryFunc>
__global__ void MapReduceKernel3_CU(TrinaryFunc mapFunc, BinaryFunc reduceFunc, T* input1, T* input2, T* input3, T* output, size_t n)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata);

   size_t blockSize = blockDim.x;
   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockSize + tid;
   size_t gridSize = blockSize*gridDim.x;
   T result = 0;

   if(i < n)
   {
      result = mapFunc.CU(input1[i], input2[i], input3[i]);
      i += gridSize;
   }

   while(i < n)
   {
      T tempMap;
      tempMap = mapFunc.CU(input1[i], input2[i], input3[i]);
      result = reduceFunc.CU(result, tempMap);
      i += gridSize;
   }

   sdata[tid] = result;

   __syncthreads();

   if(blockSize >= 512)
   {
      if (tid < 256 && tid + 256 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid + 256]);
      }
      __syncthreads();
   }
   if(blockSize >= 256)
   {
      if (tid < 128 && tid + 128 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid + 128]);
      }
      __syncthreads();
   }
   if(blockSize >= 128)
   {
      if (tid <  64 && tid +  64 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +  64]);
      }
      __syncthreads();
   }
   if(blockSize >=  64)
   {
      if (tid <  32 && tid +  32 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +  32]);
      }
      __syncthreads();
   }
   if(blockSize >=  32)
   {
      if (tid <  16 && tid +  16 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +  16]);
      }
      __syncthreads();
   }
   if(blockSize >=  16)
   {
      if (tid <   8 && tid +   8 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   8]);
      }
      __syncthreads();
   }
   if(blockSize >=   8)
   {
      if (tid <   4 && tid +   4 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   4]);
      }
      __syncthreads();
   }
   if(blockSize >=   4)
   {
      if (tid <   2 && tid +   2 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   2]);
      }
      __syncthreads();
   }
   if(blockSize >=   2)
   {
      if (tid <   1 && tid +   1 < n)
      {
         sdata[tid] = reduceFunc.CU(sdata[tid], sdata[tid +   1]);
      }
      __syncthreads();
   }

   if(tid == 0)
   {
      output[blockIdx.x] = sdata[tid];
   }
}

/*!
 *  \}
 */

}

#endif

#endif

