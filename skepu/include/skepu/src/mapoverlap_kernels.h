/*! \file mapoverlap_kernels.h
 *  \brief Contains the OpenCL and CUDA kernels for the MapOverlap skeleton.
 */

#ifndef MAPOVERLAP_KERNELS_H
#define MAPOVERLAP_KERNELS_H

#ifdef SKEPU_OPENCL

#include <string>

namespace skepu
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup MapOverlapKernels MapOverlap Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the MapOverlap skeleton.
 * \{
 */



/*!
 *
 *  OpenCL MapOverlap kernel for vector. It uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static std::string MapOverlapKernel_CL(
   "__kernel void MapOverlapKernel_KERNELNAME(__global TYPE* input, __global TYPE* output, __global TYPE* wrap, size_t n, size_t overlap, size_t out_offset, size_t out_numelements, int poly, TYPE pad, __local TYPE* sdata)\n"
   "{\n"
   "    size_t tid = get_local_id(0);\n"
   "    size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);\n"
   "    if(poly == 0)\n"
   "    {\n"
   "        sdata[overlap+tid] = (i < n) ? input[i] : pad;\n"
   "        if(tid < overlap)\n"
   "        {\n"
   "            sdata[tid] = (get_group_id(0) == 0) ? pad : input[i-overlap];\n"
   "        }\n"
   "        if(tid >= (get_local_size(0)-overlap))\n"
   "        {\n"
   "            sdata[tid+2*overlap] = (get_group_id(0) != get_num_groups(0)-1 && i+overlap < n) ? input[i+overlap] : pad;\n"
   "        }\n"
   "    }\n"
   "    else if(poly == 1)\n"
   "    {\n"
   "        if(i < n)\n"
   "        {\n"
   "           sdata[overlap+tid] = input[i];\n"
   "        }\n"
   "        else if(i-n < overlap)\n"
   "        {\n"
   "           sdata[overlap+tid] = wrap[overlap+(i-n)];\n"
   "        }\n"
   "        else\n"
   "        {\n"
   "           sdata[overlap+tid] = pad;\n"
   "        }\n"
   "        if(tid < overlap)\n"
   "        {\n"
   "               sdata[tid] = (get_group_id(0) == 0) ? wrap[tid] : input[i-overlap];\n"
   "        }\n"
   "        if(tid >= (get_local_size(0)-overlap))\n"
   "        {\n"
   "               sdata[tid+2*overlap] = (get_group_id(0) != get_num_groups(0)-1 && i+overlap < n) ? input[i+overlap] : wrap[overlap+(i+overlap-n)];\n"
   "        }\n"
   "    }\n"
   "    else if(poly == 2)\n"
   "    {\n"
   "        sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];\n"
   "        if(tid < overlap)\n"
   "        {\n"
   "            sdata[tid] = (get_group_id(0) == 0) ? input[0] : input[i-overlap];\n"
   "        }\n"
   "        if(tid >= (get_local_size(0)-overlap))\n"
   "        {\n"
   "            sdata[tid+2*overlap] = (get_group_id(0) != get_num_groups(0)-1 && i+overlap < n) ? input[i+overlap] : input[n-1];\n"
   "        }\n"
   "    }\n"
   "    barrier(CLK_LOCAL_MEM_FENCE);\n"
   "    if( (i >= out_offset) && (i < out_offset+out_numelements) )\n"
   "    {\n"
   "        output[i-out_offset] = FUNCTIONNAME(&(sdata[tid+overlap]));\n"
   "    }\n"
   "}\n"
);


/*!
 *
 *  OpenCL MapOverlap kernel for applying row-wise overlap on matrix operands. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static std::string MapOverlapKernel_CL_Matrix_Row(
   "__kernel void MapOverlapKernel_MatRowWise_KERNELNAME(__global TYPE* input, __global TYPE* output, __global TYPE* wrap, size_t n, size_t overlap, size_t out_offset, size_t out_numelements, int poly, TYPE pad, size_t blocksPerRow, size_t rowWidth, __local TYPE* sdata)\n"
   "{\n"
   "    size_t tid = get_local_id(0);\n"
   "    size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);\n"
   "    size_t wrapIndex= 2 * overlap * (int)(get_group_id(0)/blocksPerRow);\n"
   "    size_t tmp= (get_group_id(0) % blocksPerRow);\n"
   "    size_t tmp2= (get_group_id(0) / blocksPerRow);\n"
   "    if(poly == 0)\n"
   "    {\n"
   "        sdata[overlap+tid] = (i < n) ? input[i] : pad;\n"
   "        if(tid < overlap)\n"
   "        {\n"
   "            sdata[tid] = (tmp==0) ? pad : input[i-overlap];\n"
   "        }\n"
   "        if(tid >= (get_local_size(0)-overlap))\n"
   "        {\n"
   "            sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (i+overlap < n) && tmp!=(blocksPerRow-1)) ? input[i+overlap] : pad;\n"
   "        }\n"
   "    }\n"
   "    else if(poly == 1)\n"
   "    {\n"
   "        if(i < n)\n"
   "        {\n"
   "           sdata[overlap+tid] = input[i];\n"
   "        }\n"
   "        else if(i-n < overlap)\n"
   "        {\n"
   "           sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];\n"
   "        }\n"
   "        else\n"
   "        {\n"
   "           sdata[overlap+tid] = pad;\n"
   "        }\n"
   "        if(tid < overlap)\n"
   "        {\n"
   "               sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[i-overlap];\n"
   "        }\n"
   "        if(tid >= (get_local_size(0)-overlap))\n"
   "        {\n"
   "               sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && i+overlap < n && tmp!=(blocksPerRow-1)) ? input[i+overlap] : wrap[overlap+wrapIndex+(tid+overlap-get_local_size(0))];\n"
   "        }\n"
   "    }\n"
   "    else if(poly == 2)\n"
   "    {\n"
   "        sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];\n"
   "        if(tid < overlap)\n"
   "        {\n"
   "            sdata[tid] = (tmp==0) ? input[tmp2*rowWidth] : input[i-overlap];\n"
   "        }\n"
   "        if(tid >= (get_local_size(0)-overlap))\n"
   "        {\n"
   "            sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (i+overlap < n) && (tmp!=(blocksPerRow-1))) ? input[i+overlap] : input[(tmp2+1)*rowWidth-1];\n"
   "        }\n"
   "    }\n"
   "    barrier(CLK_LOCAL_MEM_FENCE);\n"
   "    if( (i >= out_offset) && (i < out_offset+out_numelements) )\n"
   "    {\n"
   "        output[i-out_offset] = FUNCTIONNAME(&(sdata[tid+overlap]));\n"
   "    }\n"
   "}\n"
);



/*!
 *
 *  OpenCL MapOverlap kernel for applying column-wise overlap on matrix operands. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static std::string MapOverlapKernel_CL_Matrix_Col(
   "__kernel void MapOverlapKernel_MatColWise_KERNELNAME(__global TYPE* input, __global TYPE* output, __global TYPE* wrap, size_t n, size_t overlap, size_t out_offset, size_t out_numelements, int poly, TYPE pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth, __local TYPE* sdata)\n"
   "{\n"
   "    size_t tid = get_local_id(0);\n"
   "    size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);\n"
   "    size_t wrapIndex= 2 * overlap * (int)(get_group_id(0)/blocksPerCol);\n"
   "    size_t tmp= (get_group_id(0) % blocksPerCol);\n"
   "    size_t tmp2= (get_group_id(0) / blocksPerCol);\n"
   "    size_t arrInd = (tid + tmp*get_local_size(0))*rowWidth + tmp2;\n"
   "    if(poly == 0)\n"
   "    {\n"
   "        sdata[overlap+tid] = (i < n) ? input[arrInd] : pad;\n"
   "        if(tid < overlap)\n"
   "        {\n"
   "            sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];\n"
   "        }\n"
   "        if(tid >= (get_local_size(0)-overlap))\n"
   "        {\n"
   "            sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : pad;\n"
   "        }\n"
   "    }\n"
   "    else if(poly == 1)\n"
   "    {\n"
   "        if(i < n)\n"
   "        {\n"
   "           sdata[overlap+tid] = input[arrInd];\n"
   "        }\n"
   "        else if(i-n < overlap)\n"
   "        {\n"
   "           sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];\n"
   "        }\n"
   "        else\n"
   "        {\n"
   "           sdata[overlap+tid] = pad;\n"
   "        }\n"
   "        if(tid < overlap)\n"
   "        {\n"
   "               sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[(arrInd-(overlap*rowWidth))];\n"
   "        }\n"
   "        if(tid >= (get_local_size(0)-overlap))\n"
   "        {\n"
   "               sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : wrap[overlap+wrapIndex+(tid+overlap-get_local_size(0))];\n"
   "        }\n"
   "    }\n"
   "    else if(poly == 2)\n"
   "    {\n"
   "        sdata[overlap+tid] = (i < n) ? input[arrInd] : input[n-1];\n"
   "        if(tid < overlap)\n"
   "        {\n"
   "            sdata[tid] = (tmp==0) ? input[tmp2] : input[(arrInd-(overlap*rowWidth))];\n"
   "        }\n"
   "        if(tid >= (get_local_size(0)-overlap))\n"
   "        {\n"
   "            sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : input[tmp2+(colWidth-1)*rowWidth];\n"
   "        }\n"
   "    }\n"
   "    barrier(CLK_LOCAL_MEM_FENCE);\n"
   "    if( (arrInd >= out_offset) && (arrInd < out_offset+out_numelements) )\n"
   "    {\n"
   "        output[arrInd-out_offset] = FUNCTIONNAME(&(sdata[tid+overlap]));\n"
   "    }\n"
   "}\n"
);



/*!
 *
 *  OpenCL MapOverlap kernel for applying column-wise overlap on matrix operands when using multiple GPUs. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
static std::string MapOverlapKernel_CL_Matrix_ColMulti(
   "__kernel void MapOverlapKernel_MatColWiseMulti_KERNELNAME(__global TYPE* input, __global TYPE* output, __global TYPE* wrap, size_t n, size_t overlap, size_t in_offset, size_t out_numelements, int poly, int deviceType, TYPE pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth, __local TYPE* sdata)\n"
   "{\n"
   "    size_t tid = get_local_id(0);\n"
   "    size_t i = get_group_id(0) * get_local_size(0) + get_local_id(0);\n"
   "    size_t wrapIndex= 2 * overlap * (int)(get_group_id(0)/blocksPerCol);\n"
   "    size_t tmp= (get_group_id(0) % blocksPerCol);\n"
   "    size_t tmp2= (get_group_id(0) / blocksPerCol);\n"
   "    size_t arrInd = (tid + tmp*get_local_size(0))*rowWidth + tmp2;\n"
   "    if(poly == 0)\n"
   "    {\n"
   "       sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : pad;\n"
   "		if(deviceType == -1)\n"
   "    	{\n"
   "	        if(tid < overlap)\n"
   "	        {\n"
   "	            sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];\n"
   "	        }\n"
   "			 \n"
   "	        if(tid >= (get_local_size(0)-overlap))\n"
   "	        {\n"
   "	            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];\n"
   "	        }\n"
   "    	}\n"
   "    	else if(deviceType == 0) \n"
   "    	{\n"
   "    		if(tid < overlap)\n"
   "	        {\n"
   "       	    sdata[tid] = input[arrInd];\n"
   "	        }\n"
   "	        if(tid >= (get_local_size(0)-overlap))\n"
   "	        {\n"
   "				sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];\n"
   "	        }\n"
   "    	}\n"
   "    	else if(deviceType == 1)\n"
   "    	{\n"
   "    		if(tid < overlap)\n"
   "	        {\n"
   "       	    sdata[tid] = input[arrInd];\n"
   "	        }\n"
   "	        if(tid >= (get_local_size(0)-overlap))\n"
   "	        {\n"
   "				sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : pad;\n"
   "	        }\n"
   "    	}\n"
   "    }\n"
   "    else if(poly == 1)\n"
   "    {\n"
   "		sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : ((i-n < overlap) ? wrap[(i-n)+ (overlap * tmp2)] : pad);\n"
   "		if(deviceType == -1)\n"
   "		{\n"
   "	        if(tid < overlap)\n"
   "	        {\n"
   "		    	sdata[tid] = (tmp==0) ? wrap[tid+(overlap * tmp2)] : input[(arrInd-(overlap*rowWidth))];\n"
   "	        }\n"
   "	        if(tid >= (get_local_size(0)-overlap))\n"
   "	        {\n"
   "			    sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];\n"
   "	        }\n"
   "		}\n"
   "		else if(deviceType == 0)\n"
   "		{\n"
   "	        if(tid < overlap)\n"
   "	        {\n"
   "		    	sdata[tid] = input[arrInd];\n"
   "	        }\n"
   "	        if(tid >= (get_local_size(0)-overlap))\n"
   "	        {\n"
   "	        	sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];\n"
   "	        }\n"
   "		}\n"
   "		else if(deviceType == 1)\n"
   "		{\n"
   "	        if(tid < overlap)\n"
   "	        {\n"
   "	        	sdata[tid] = input[arrInd];\n"
   "	        }\n"
   "	        if(tid >= (get_local_size(0)-overlap))\n"
   "	        {\n"
   "			    sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : wrap[(overlap * tmp2)+(tid+overlap-get_local_size(0))];\n"
   "	        }\n"
   "		}\n"
   "    }\n"
   "    else if(poly == 2)\n"
   "    {\n"
   "		sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : input[n+in_offset-1];\n"
   "		if(deviceType == -1)\n"
   "		{\n"
   "	        if(tid < overlap)\n"
   "	        {\n"
   "			    sdata[tid] = (tmp==0) ? input[tmp2] : input[(arrInd-(overlap*rowWidth))];\n"
   "	        }\n"
   "	        if(tid >= (get_local_size(0)-overlap))\n"
   "	        {\n"
   "		    	sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];\n"
   "	        }\n"
   "		}\n"
   "		else if(deviceType == 0)\n"
   "		{\n"
   "	        if(tid < overlap)\n"
   "	        {\n"
   "			    sdata[tid] = input[arrInd];\n"
   "	        }\n"
   "	        if(tid >= (get_local_size(0)-overlap))\n"
   "	        {\n"
   "				sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];\n"
   "	        }\n"
   "		}\n"
   "		else if(deviceType == 1)\n"
   "		{\n"
   "	        if(tid < overlap)\n"
   "	        {\n"
   "			    sdata[tid] = input[arrInd];\n"
   "	        }\n"
   "	        if(tid >= (get_local_size(0)-overlap))\n"
   "	        {\n"
   "		    	sdata[tid+2*overlap] = (get_group_id(0) != (get_num_groups(0)-1) && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : input[tmp2+in_offset+(colWidth-1)*rowWidth];\n"
   "	        }\n"
   "		}\n"
   "    }\n"
   "    barrier(CLK_LOCAL_MEM_FENCE);\n"
   "    if( arrInd < out_numelements )\n"
   "    {\n"
   "        output[arrInd] = FUNCTIONNAME(&(sdata[tid+overlap]));\n"
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
 *  \defgroup MapOverlapKernels MapOverlap Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the MapOverlap skeleton.
 * \{
 */

/*!
* Matrix transpose CUDA kernel
*/
template <typename T>
__global__ void transpose(T *odata, T *idata, size_t width, size_t height)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata);

   size_t block_dim= blockDim.x;
   size_t block_dimY= blockDim.y;
   // read the matrix tile into shared memory
   size_t xIndex = blockIdx.x * block_dim + threadIdx.x;
   size_t yIndex = blockIdx.y * block_dimY + threadIdx.y;
   if((xIndex < width) && (yIndex < height))
   {
      size_t index_in = yIndex * width + xIndex;
      sdata[threadIdx.y][threadIdx.x] = idata[index_in];
   }

   __syncthreads();

   // write the transposed matrix tile to global memory
   xIndex = blockIdx.y * block_dim + threadIdx.x;
   yIndex = blockIdx.x * block_dimY + threadIdx.y;
   if((xIndex < height) && (yIndex < width))
   {
      size_t index_out = yIndex * height + xIndex;
      odata[index_out] = sdata[threadIdx.x][threadIdx.y];
   }
}

/*!
 *  CUDA MapOverlap kernel for vector operands. It uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
template <int poly, typename T, typename OverlapFunc>
__global__ void MapOverlapKernel_CU(OverlapFunc mapOverlapFunc, T* input, T* output, T* wrap, size_t n, size_t out_offset, size_t out_numelements, T pad)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   size_t overlap = mapOverlapFunc.overlap;

   while(i<(n+overlap-1))
   {
      //Copy data to shared memory
      if(poly == 0) // constant policy
      {
         sdata[overlap+tid] = (i < n) ? input[i] : pad;

         if(tid < overlap)
         {
            sdata[tid] = (i<overlap) ? pad : input[i-overlap];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (i+overlap < n) ? input[i+overlap] : pad;
         }
      }
      else if(poly == 1)
      {
         if(i < n)
         {
            sdata[overlap+tid] = input[i];
         }
         else
         {
            sdata[overlap+tid] = wrap[overlap+(i-n)];
         }

         if(tid < overlap)
         {
            sdata[tid] = (i<overlap) ? wrap[tid] : input[i-overlap];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (i+overlap < n) ? input[i+overlap] : wrap[overlap+(i+overlap-n)];
         }
      }
      else if(poly == 2) // DUPLICATE
      {
         sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];

         if(tid < overlap)
         {
            sdata[tid] = (i<overlap) ? input[0] : input[i-overlap];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (i+overlap < n) ? input[i+overlap] : input[n-1];
         }
      }

      __syncthreads();

      //Compute and store data
      if( (i >= out_offset) && (i < out_offset+out_numelements) )
      {
         output[i-out_offset] = mapOverlapFunc.CU(&(sdata[tid+overlap]));
      }

      i += gridSize;

      __syncthreads();
   }
}



/*!
 *
 *  CUDA MapOverlap kernel for applying row-wise overlap on matrix operands. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
template <int poly, typename T, typename OverlapFunc>
__global__ void MapOverlapKernel_CU_Matrix_Row(OverlapFunc mapOverlapFunc, T* input, T* output, T* wrap, size_t n, size_t out_offset, size_t out_numelements, T pad, size_t blocksPerRow, size_t rowWidth)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + tid;
   size_t overlap = mapOverlapFunc.overlap;

   size_t wrapIndex= 2 * overlap * (int)(blockIdx.x/blocksPerRow);
   size_t tmp= (blockIdx.x % blocksPerRow);
   size_t tmp2= (blockIdx.x / blocksPerRow);


   //Copy data to shared memory
   if(poly == 0)
   {
      sdata[overlap+tid] = (i < n) ? input[i] : pad;

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? pad : input[i-overlap];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (i+overlap < n) && tmp!=(blocksPerRow-1)) ? input[i+overlap] : pad;
      }
   }
   else if(poly == 1)
   {
      if(i < n)
      {
         sdata[overlap+tid] = input[i];
      }
      else if(i-n < overlap)
      {
         sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];
      }
      else
      {
         sdata[overlap+tid] = pad;
      }

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[i-overlap];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && i+overlap < n && tmp!=(blocksPerRow-1)) ? input[i+overlap] : wrap[overlap+wrapIndex+(tid+overlap-blockDim.x)];
      }
   }
   else if(poly == 2)
   {
      sdata[overlap+tid] = (i < n) ? input[i] : input[n-1];

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? input[tmp2*rowWidth] : input[i-overlap];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (i+overlap < n) && (tmp!=(blocksPerRow-1))) ? input[i+overlap] : input[(tmp2+1)*rowWidth-1];
      }
   }

   __syncthreads();

   //Compute and store data
   if( (i >= out_offset) && (i < out_offset+out_numelements) )
   {
      output[i-out_offset] = mapOverlapFunc.CU(&(sdata[tid+overlap]));
   }
}



/*!
 *
 *  CUDA MapOverlap kernel for applying column-wise overlap on matrix operands. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 */
template <int poly, typename T, typename OverlapFunc>
__global__ void MapOverlapKernel_CU_Matrix_Col(OverlapFunc mapOverlapFunc, T* input, T* output, T* wrap, size_t n, size_t out_offset, size_t out_numelements, T pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + tid;
   size_t overlap = mapOverlapFunc.overlap;

   size_t wrapIndex= 2 * overlap * (int)(blockIdx.x/blocksPerCol);
   size_t tmp= (blockIdx.x % blocksPerCol);
   size_t tmp2= (blockIdx.x / blocksPerCol);

   size_t arrInd = (threadIdx.x + tmp*blockDim.x)*rowWidth + ((blockIdx.x)/blocksPerCol);

   //Copy data to shared memory
   if(poly == 0)
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd] : pad;

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : pad;
      }
   }
   else if(poly == 1)
   {
      if(i < n)
      {
         sdata[overlap+tid] = input[arrInd];
      }
      else if(i-n < overlap)
      {
         sdata[overlap+tid] = wrap[(overlap+(i-n))+ wrapIndex];
      }
      else
      {
         sdata[overlap+tid] = pad;
      }

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? wrap[tid+wrapIndex] : input[(arrInd-(overlap*rowWidth))];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : wrap[overlap+wrapIndex+(tid+overlap-blockDim.x)];
      }
   }
   else if(poly == 2)
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd] : input[n-1];

      if(tid < overlap)
      {
         sdata[tid] = (tmp==0) ? input[tmp2] : input[(arrInd-(overlap*rowWidth))];
      }

      if(tid >= (blockDim.x-overlap))
      {
         sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+(overlap*rowWidth))] : input[tmp2+(colWidth-1)*rowWidth];
      }
   }

   __syncthreads();

   //Compute and store data
   if( (arrInd >= out_offset) && (arrInd < out_offset+out_numelements) )
   {
      output[arrInd-out_offset] = mapOverlapFunc.CU(&(sdata[tid+overlap]));
   }
}







/*!
 *
 *  CUDA MapOverlap kernel for applying column-wise overlap on matrix operands with multiple devices. It
 *  uses one device thread per element so maximum number of device threads
 *  limits the maximum number of elements this kernel can handle. Also the amount of shared memory and
 *  the maximum blocksize limits the maximum overlap that is possible to use, typically limits the
 *  overlap to < 256.
 *  Device type : -1 (first), 0 (middleDevice), 1(last),
 */
template <int poly, int deviceType, typename T, typename OverlapFunc>
__global__ void MapOverlapKernel_CU_Matrix_ColMulti(OverlapFunc mapOverlapFunc, T* input, T* output, T* wrap, size_t n, size_t in_offset, size_t out_numelements, T pad, size_t blocksPerCol, size_t rowWidth, size_t colWidth)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata);

   size_t tid = threadIdx.x;
   size_t i = blockIdx.x * blockDim.x + tid;
   size_t overlap = mapOverlapFunc.overlap;

   size_t tmp= (blockIdx.x % blocksPerCol);
   size_t tmp2= (blockIdx.x / blocksPerCol);

   size_t arrInd = (threadIdx.x + tmp*blockDim.x)*rowWidth + tmp2; //((blockIdx.x)/blocksPerCol);

   if(poly == 0) //IF overlap policy is CONSTANT
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : pad; // in_offset

      if(deviceType == -1) // first device, i.e. in_offset=0
      {
         if(tid < overlap)
         {
            sdata[tid] = (tmp==0) ? pad : input[(arrInd-(overlap*rowWidth))];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 0) // middle device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 1) // last device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+in_offset+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : pad;
         }
      }
   }
   else if(poly == 1) //IF overlap policy is CYCLIC
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : ((i-n < overlap) ? wrap[(i-n)+ (overlap * tmp2)] : pad);

      if(deviceType == -1) // first device, i.e. in_offset=0
      {
         if(tid < overlap)
         {
            sdata[tid] = (tmp==0) ? wrap[tid+(overlap * tmp2)] : input[(arrInd-(overlap*rowWidth))];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 0) // middle device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 1) // last device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : wrap[(overlap * tmp2)+(tid+overlap-blockDim.x)];
         }
      }
   }
   else if(poly == 2) //IF overlap policy is DUPLICATE
   {
      sdata[overlap+tid] = (i < n) ? input[arrInd+in_offset] : input[n+in_offset-1];

      if(deviceType == -1) // first device, i.e. in_offset=0
      {
         if(tid < overlap)
         {
            sdata[tid] = (tmp==0) ? input[tmp2] : input[(arrInd-(overlap*rowWidth))];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 0) // middle device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = input[(arrInd+in_offset+(overlap*rowWidth))];
         }
      }
      else if(deviceType == 1) // last device
      {
         if(tid < overlap)
         {
            sdata[tid] = input[arrInd];
         }

         if(tid >= (blockDim.x-overlap))
         {
            sdata[tid+2*overlap] = (blockIdx.x != gridDim.x-1 && (arrInd+in_offset+(overlap*rowWidth)) < n && (tmp!=(blocksPerCol-1))) ? input[(arrInd+in_offset+(overlap*rowWidth))] : input[tmp2+in_offset+(colWidth-1)*rowWidth];
         }
      }
   }

   __syncthreads();

   //Compute and store data
   if( arrInd < out_numelements )
   {
      output[arrInd] = mapOverlapFunc.CU(&(sdata[tid+overlap]));
   }
}


/*!
 *  \}
 */

}

#endif

#endif

