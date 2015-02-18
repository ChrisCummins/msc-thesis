/*! \file reduce_kernels.h
 *  \brief Contains the OpenCL and CUDA kernels for the Reduce skeleton (used for both 1D and 2D reduce operation).
 */

#ifndef REDUCE_KERNELS_H
#define REDUCE_KERNELS_H

#ifdef SKEPU_OPENCL

#include <string>

namespace skepu
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup ReduceKernels Reduce Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Reduce skeleton.
 * \{
 */

/*!
 *
 *  OpenCL Reduce kernel, using the same pattern as reduce6 in the CUDA SDK. See whitepaper from NVIDIA on optimizing
 *  reduction for the GPU.
 */
static std::string ReduceKernel_CL(
   "__kernel void ReduceKernel_KERNELNAME(__global TYPE* input, __global TYPE* output, size_t n, __local TYPE* sdata)\n"
   "{\n"
   "    size_t blockSize = get_local_size(0);\n"
   "    size_t tid = get_local_id(0);\n"
   "    size_t i = get_group_id(0)*blockSize + get_local_id(0);\n"
   "    size_t gridSize = blockSize*get_num_groups(0);\n"
   "    TYPE result = 0;\n"
   "    if(i < n)\n"
   "    {\n"
   "        result = input[i];\n"
   "        i += gridSize;\n"
   "    }\n"
   "    while(i < n)\n"
   "    {\n"
   "        result = FUNCTIONNAME(result, input[i], (TYPE)0);\n"
   "        i += gridSize;\n"
   "    }\n"
   "    sdata[tid] = result;\n"
   "    barrier(CLK_LOCAL_MEM_FENCE);\n"
   "    if(blockSize >= 512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = FUNCTIONNAME(sdata[tid], sdata[tid + 256], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >= 256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = FUNCTIONNAME(sdata[tid], sdata[tid + 128], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >= 128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = FUNCTIONNAME(sdata[tid], sdata[tid +  64], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = FUNCTIONNAME(sdata[tid], sdata[tid +  32], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = FUNCTIONNAME(sdata[tid], sdata[tid +  16], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=  16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = FUNCTIONNAME(sdata[tid], sdata[tid +   8], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = FUNCTIONNAME(sdata[tid], sdata[tid +   4], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = FUNCTIONNAME(sdata[tid], sdata[tid +   2], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(blockSize >=   2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = FUNCTIONNAME(sdata[tid], sdata[tid +   1], (TYPE)0); } barrier(CLK_LOCAL_MEM_FENCE); }\n"
   "    if(tid == 0)\n"
   "    {\n"
   "        output[get_group_id(0)] = sdata[tid];\n"
   "    }\n"
   "}\n"
);


/*!
 * \brief A helper to return a value that is nearest value that is power of 2.
 *
 * \param x The input number for which we need to find the nearest value that is power of 2.
 * \return The nearest value that is power of 2.
 */
size_t nextPow2( size_t x )
{
   --x;
   x |= x >> 1;
   x |= x >> 2;
   x |= x >> 4;
   x |= x >> 8;
   x |= x >> 16;
   return ++x;
}



/*!
 * Compute the number of threads and blocks to use for the reduction kernel.
 * We set threads / block to the minimum of maxThreads and n/2 where n is
 * problem size. We observe the maximum specified number of blocks, because
 * each kernel thread can process more than 1 elements.
 *
 * \param n Problem size.
 * \param maxBlocks Maximum number of blocks that can be used.
 * \param maxThreads Maximum number of threads that can be used.
 * \param blocks An output parameter passed by reference. Specify number of blocks to be used.
 * \param threads An output parameter passed by reference. Specify number of threads to be used.
 */
void getNumBlocksAndThreads(size_t n, size_t maxBlocks, size_t maxThreads, size_t &blocks, size_t &threads)
{
   threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
   blocks = (n + (threads * 2 - 1)) / (threads * 2);

   blocks = MIN(maxBlocks, blocks);
}
/*!
 *  \}
 */

}

#endif

#ifdef SKEPU_CUDA


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
   __device__ inline operator       T*()
   {
      extern __shared__ int __smem[];
      return (T*)__smem;
   }

   __device__ inline operator const T*() const
   {
      extern __shared__ int __smem[];
      return (T*)__smem;
   }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
   __device__ inline operator       double*()
   {
      extern __shared__ double __smem_d[];
      return (double*)__smem_d;
   }

   __device__ inline operator const double*() const
   {
      extern __shared__ double __smem_d[];
      return (double*)__smem_d;
   }
};

namespace skepu
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup ReduceKernels Reduce Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Reduce skeleton.
 * \{
 */

/*!
 *
 *  The old CUDA Reduce kernel which now gives incorrect results for larger problem sizes. Not used anymore.
 */
template<typename T, typename BinaryFunc>
__global__ void ReduceKernel_CU_oldAndIncorrect(BinaryFunc reduceFunc, T* input, T* output, size_t n)
{
   T* sdata = SharedMemory<T>();

   size_t blockSize = blockDim.x;
   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockSize + tid;
   size_t gridSize = blockSize*gridDim.x;
   T result = 0;

   if(i < n)
   {
      result = input[i];
      i += gridSize;
   }

   while(i < n)
   {
      result = reduceFunc.CU(result, input[i]);
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
 *  CUDA Reduce kernel, using the same pattern as reduce6 in the CUDA SDK. See whitepaper from NVIDIA on optimizing
 *  reduction for the GPU.
 */
template<typename T, typename BinaryFunc, size_t blockSize, bool nIsPow2>
__global__ void ReduceKernel_CU(BinaryFunc reduceFunc, T *input, T *output, size_t n)
{
   T *sdata = SharedMemory<T>();

   // perform first level of reduction,
   // reading from global memory, writing to shared memory
   size_t tid = threadIdx.x;
   size_t i = blockIdx.x*blockSize*2 + threadIdx.x;
   size_t gridSize = blockSize*2*gridDim.x;

   T result = 0;

   if(i < n)
   {
      result = input[i];
      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
      //This nIsPow2 opt is not valid when we use this kernel for sparse matrices as well where we
      // dont exactly now the elements when calculating thread- and block-size and nIsPow2 assum becomes invalid in some cases there which results in sever problems.
      // There we pass it always false
      if (nIsPow2 || i + blockSize < n)
         result = reduceFunc.CU(result, input[i+blockSize]);
      i += gridSize;
   }

   // we reduce multiple elements per thread.  The number is determined by the
   // number of active thread blocks (via gridDim).  More blocks will result
   // in a larger gridSize and therefore fewer elements per thread
   while(i < n)
   {
      result = reduceFunc.CU(result, input[i]);
      // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
      if (nIsPow2 || i + blockSize < n)
         result = reduceFunc.CU(result, input[i+blockSize]);
      i += gridSize;
   }

   // each thread puts its local sum into shared memory
   sdata[tid] = result;

   __syncthreads();


   // do reduction in shared mem
   if (blockSize >= 512)
   {
      if (tid < 256)
      {
         sdata[tid] = result = reduceFunc.CU(result, sdata[tid + 256]);
      }
      __syncthreads();
   }
   if (blockSize >= 256)
   {
      if (tid < 128)
      {
         sdata[tid] = result = reduceFunc.CU(result, sdata[tid + 128]);
      }
      __syncthreads();
   }
   if (blockSize >= 128)
   {
      if (tid <  64)
      {
         sdata[tid] = result = reduceFunc.CU(result, sdata[tid +  64]);
      }
      __syncthreads();
   }

   if (tid < 32)
   {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile T* smem = sdata;
      if (blockSize >=  64)
      {
         smem[tid] = result = reduceFunc.CU(result, smem[tid + 32]);
      }
      if (blockSize >=  32)
      {
         smem[tid] = result = reduceFunc.CU(result, smem[tid + 16]);
      }
      if (blockSize >=  16)
      {
         smem[tid] = result = reduceFunc.CU(result, smem[tid +  8]);
      }
      if (blockSize >=   8)
      {
         smem[tid] = result = reduceFunc.CU(result, smem[tid +  4]);
      }
      if (blockSize >=   4)
      {
         smem[tid] = result = reduceFunc.CU(result, smem[tid +  2]);
      }
      if (blockSize >=   2)
      {
         smem[tid] = result = reduceFunc.CU(result, smem[tid +  1]);
      }
   }

   // write result for this block to global mem
   if (tid == 0)
      output[blockIdx.x] = sdata[0];
}





// ********************************************************************************************************************
// --------------------------------------------------------------------------------------------------------------------
// ********************************************************************************************************************
// --------------------------------------------------------------------------------------------------------------------// ********************************************************************************************************************
// --------------------------------------------------------------------------------------------------------------------// ********************************************************************************************************************
// --------------------------------------------------------------------------------------------------------------------



/*!
 * \brief A small helper to determine whether the number is a power of 2.
 *
 * \param x the actual number.
 * \return bool specifying whether number of power of 2 or not,
 */
bool isPow2(size_t x)
{
   return ((x&(x-1))==0);
}


/*!
 * Helper method used to call the actual CUDA kernel for reduction. Used when PINNED MEMORY is disabled
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param size size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template <typename ReduceFunc, typename T>
void CallReduceKernel(ReduceFunc *reduceFunc, size_t size, size_t numThreads, size_t numBlocks, T *d_idata, T *d_odata, bool enableIsPow2=true)
{
   dim3 dimBlock(numThreads, 1, 1);
   dim3 dimGrid(numBlocks, 1, 1);

   // when there is only one warp per block, we need to allocate two warps
   // worth of shared memory so that we don't index shared memory out of bounds
   size_t smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);

   // choose which of the optimized versions of reduction to launch
   if (isPow2(size) && enableIsPow2)
   {
      switch (numThreads)
      {
      case 512:
         ReduceKernel_CU<T, ReduceFunc, 512, true><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 256:
         ReduceKernel_CU<T, ReduceFunc, 256, true><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 128:
         ReduceKernel_CU<T, ReduceFunc, 128, true><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 64:
         ReduceKernel_CU<T, ReduceFunc,  64, true><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 32:
         ReduceKernel_CU<T, ReduceFunc,  32, true><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 16:
         ReduceKernel_CU<T, ReduceFunc,  16, true><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  8:
         ReduceKernel_CU<T, ReduceFunc,   8, true><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  4:
         ReduceKernel_CU<T, ReduceFunc,   4, true><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  2:
         ReduceKernel_CU<T, ReduceFunc,   2, true><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  1:
         ReduceKernel_CU<T, ReduceFunc,   1, true><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      }
   }
   else
   {
      switch (numThreads)
      {
      case 512:
         ReduceKernel_CU<T, ReduceFunc, 512, false><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 256:
         ReduceKernel_CU<T, ReduceFunc, 256, false><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 128:
         ReduceKernel_CU<T, ReduceFunc, 128, false><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 64:
         ReduceKernel_CU<T, ReduceFunc,  64, false><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 32:
         ReduceKernel_CU<T, ReduceFunc,  32, false><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 16:
         ReduceKernel_CU<T, ReduceFunc,  16, false><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  8:
         ReduceKernel_CU<T, ReduceFunc,   8, false><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  4:
         ReduceKernel_CU<T, ReduceFunc,   4, false><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  2:
         ReduceKernel_CU<T, ReduceFunc,   2, false><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  1:
         ReduceKernel_CU<T, ReduceFunc,   1, false><<< dimGrid, dimBlock, smemSize >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      }
   }
}




#ifdef USE_PINNED_MEMORY
/*!
 * Helper method used to call the actual CUDA kernel for reduction. Used when PINNED MEMORY is enabled
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param size size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param stream CUDA stream to be used.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template <typename ReduceFunc, typename T>
void CallReduceKernel_WithStream(ReduceFunc *reduceFunc, size_t size, size_t numThreads, size_t numBlocks, T *d_idata, T *d_odata, cudaStream_t &stream, bool enableIsPow2=true)
{
   dim3 dimBlock(numThreads, 1, 1);
   dim3 dimGrid(numBlocks, 1, 1);

   // when there is only one warp per block, we need to allocate two warps
   // worth of shared memory so that we don't index shared memory out of bounds
   size_t smemSize = (numThreads <= 32) ? 2 * numThreads * sizeof(T) : numThreads * sizeof(T);

   // choose which of the optimized versions of reduction to launch
   if (isPow2(size) && enableIsPow2)
   {
      switch (numThreads)
      {
      case 512:
         ReduceKernel_CU<T, ReduceFunc, 512, true><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 256:
         ReduceKernel_CU<T, ReduceFunc, 256, true><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 128:
         ReduceKernel_CU<T, ReduceFunc, 128, true><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 64:
         ReduceKernel_CU<T, ReduceFunc,  64, true><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 32:
         ReduceKernel_CU<T, ReduceFunc,  32, true><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 16:
         ReduceKernel_CU<T, ReduceFunc,  16, true><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  8:
         ReduceKernel_CU<T, ReduceFunc,   8, true><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  4:
         ReduceKernel_CU<T, ReduceFunc,   4, true><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  2:
         ReduceKernel_CU<T, ReduceFunc,   2, true><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  1:
         ReduceKernel_CU<T, ReduceFunc,   1, true><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      }
   }
   else
   {
      switch (numThreads)
      {
      case 512:
         ReduceKernel_CU<T, ReduceFunc, 512, false><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 256:
         ReduceKernel_CU<T, ReduceFunc, 256, false><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 128:
         ReduceKernel_CU<T, ReduceFunc, 128, false><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 64:
         ReduceKernel_CU<T, ReduceFunc,  64, false><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 32:
         ReduceKernel_CU<T, ReduceFunc,  32, false><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case 16:
         ReduceKernel_CU<T, ReduceFunc,  16, false><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  8:
         ReduceKernel_CU<T, ReduceFunc,   8, false><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  4:
         ReduceKernel_CU<T, ReduceFunc,   4, false><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  2:
         ReduceKernel_CU<T, ReduceFunc,   2, false><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      case  1:
         ReduceKernel_CU<T, ReduceFunc,   1, false><<< dimGrid, dimBlock, smemSize, stream >>>(*reduceFunc, d_idata, d_odata, size);
         break;
      }
   }
}
#endif




/*!
 * \brief A helper to return a value that is nearest value that is power of 2.
 *
 * \param x The input number for which we need to find the nearest value that is power of 2.
 * \return The nearest value that is power of 2.
 */
size_t nextPow2( size_t x )
{
   --x;
   x |= x >> 1;
   x |= x >> 2;
   x |= x >> 4;
   x |= x >> 8;
   x |= x >> 16;
   return ++x;
}


/*!
 * Compute the number of threads and blocks to use for the reduction kernel.
 * We set threads / block to the minimum of maxThreads and n/2 where n is
 * problem size. We observe the maximum specified number of blocks, because
 * each kernel thread can process more than 1 elements.
 *
 * \param n Problem size.
 * \param maxBlocks Maximum number of blocks that can be used.
 * \param maxThreads Maximum number of threads that can be used.
 * \param blocks An output parameter passed by reference. Specify number of blocks to be used.
 * \param threads An output parameter passed by reference. Specify number of threads to be used.
 */
void getNumBlocksAndThreads(size_t n, size_t maxBlocks, size_t maxThreads, size_t &blocks, size_t &threads)
{
   threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
   blocks = (n + (threads * 2 - 1)) / (threads * 2);

   blocks = MIN(maxBlocks, blocks);
}




#ifdef USE_PINNED_MEMORY
/*!
 *  A helper function that is used to call the actual kernel for reduction. Used by other functions to call the actual kernel
 *  Internally, it just calls 2 kernels by setting their arguments. No synchronization is enforced.
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param n size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param maxThreads Maximum number of threads that can be used for kernel execution.
 *  \param maxBlocks Maximum number of blocks that can be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param stream CUDA stream to be used, only when using Pinned memory allocations.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template <typename ReduceFunc, typename T>
void ExecuteReduceOnADevice(ReduceFunc *reduceFunc, size_t  n, size_t  numThreads, size_t  numBlocks, size_t  maxThreads, size_t  maxBlocks, T* d_idata, T* d_odata, unsigned int deviceID, cudaStream_t &stream, bool enableIsPow2=true)
#else
/*!
 *  A helper function that is used to call the actual kernel for reduction. Used by other functions to call the actual kernel
 *  Internally, it just calls 2 kernels by setting their arguments. No synchronization is enforced.
 *
 *  \param reduceFunc The reduction user function to be used.
 *  \param n size of the input array to be reduced.
 *  \param numThreads Number of threads to be used for kernel execution.
 *  \param numBlocks Number of blocks to be used for kernel execution.
 *  \param maxThreads Maximum number of threads that can be used for kernel execution.
 *  \param maxBlocks Maximum number of blocks that can be used for kernel execution.
 *  \param d_idata CUDA memory pointer to input array.
 *  \param d_odata CUDA memory pointer to output array.
 *  \param deviceID Integer deciding which device to utilize.
 *  \param enableIsPow2 boolean flag (default true) used to enable/disable isPow2 optimizations. disabled only for sparse row-/column-wise reduction for technical reasons.
 */
template <typename ReduceFunc, typename T>
void ExecuteReduceOnADevice(ReduceFunc *reduceFunc, size_t  n, size_t  numThreads, size_t  numBlocks, size_t  maxThreads, size_t  maxBlocks, T* d_idata, T* d_odata, unsigned int deviceID, bool enableIsPow2=true)
#endif
{
   // execute the kernel
#ifdef USE_PINNED_MEMORY
   CallReduceKernel_WithStream<ReduceFunc, T>(reduceFunc, n, numThreads, numBlocks, d_idata, d_odata, stream, enableIsPow2);
#else
   CallReduceKernel<ReduceFunc, T>(reduceFunc, n, numThreads, numBlocks, d_idata, d_odata, enableIsPow2);
#endif

   // sum partial block sums on GPU
   size_t s=numBlocks;

   while(s > 1)
   {
      size_t threads = 0, blocks = 0;
      getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

#ifdef USE_PINNED_MEMORY
      CallReduceKernel_WithStream<ReduceFunc, T>(reduceFunc, s, threads, blocks, d_odata, d_odata, stream, enableIsPow2);
#else
      CallReduceKernel<ReduceFunc, T>(reduceFunc, s, threads, blocks, d_odata, d_odata, enableIsPow2);
#endif

      s = (s + (threads*2-1)) / (threads*2);
   }
}



/*!
 *  \}
 */

}

#endif



#endif

