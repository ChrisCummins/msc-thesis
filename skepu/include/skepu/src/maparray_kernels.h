/*! \file maparray_kernels.h
 *  \brief Contains the OpenCL and CUDA kernels for the MapArray skeleton.
 */

#ifndef MAPARRAY_KERNELS_H
#define MAPARRAY_KERNELS_H

#ifdef SKEPU_OPENCL

#include <string>

namespace skepu
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup MapArrayKernels MapArray Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Map skeleton.
 * \{
 */

/*!
 *
 *  OpenCL MapArray kernel for vector operands. Similar to Map computation but a pointer to first vector is supplied to the user function instead
 *  of just one element.
 */
static std::string MapArrayKernel_CL(
   "__kernel void MapArrayKernel_KERNELNAME(__global TYPE* input1, __global TYPE* input2, __global TYPE* output, size_t n, CONST_TYPE const1)\n"
   "{\n"
   "    size_t i = get_global_id(0);\n"
   "    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"
   "    while(i < n)\n"
   "    {\n"
   "        output[i] = FUNCTIONNAME(&input1[0], input2[i], const1);\n"
   "        i += gridSize;\n"
   "    }\n"
   "}\n"
);





/*!
 *
 *  OpenCL MapArray kernel for Vector-Matrix that applies array block-wise. Similar to Map computation but a pointer to first vector is supplied to the user function instead
 *  of just one element.
 */
static std::string MapArrayKernel_CL_Matrix_Blockwise(
   "__kernel void MapArrayKernel_Matrix_Blockwise_KERNELNAME(__global TYPE* input1, __global TYPE* input2, __global TYPE* output, size_t outSize, size_t p2BlockSize, CONST_TYPE const1)\n"
   "{\n"
   "    size_t i = get_global_id(0);\n"
   "    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"
   "    if(i < outSize)\n"
   "    {\n"
   "        output[i] = FUNCTIONNAME(&input1[0], &input2[i*p2BlockSize], const1);\n"
   "        i += gridSize;\n"
   "    }\n"
   "}\n"
);


/*!
 *
 *  OpenCL MapArray kernel for Vector-SparseMatrix that applies array block-wise. Similar to Map computation but a pointer to first vector is supplied to the user function instead
 *  of just one element.
 */
static std::string MapArrayKernel_CL_Sparse_Matrix_Blockwise(
   "__kernel void MapArrayKernel_Sparse_Matrix_Blockwise_KERNELNAME(__global TYPE* input1, __global TYPE* in2_values, __global size_t *in2_row_offsets, __global size_t *in2_col_indices, __global TYPE* output, size_t outSize, size_t indexOffset, CONST_TYPE const1)\n"
   "{\n"
   "    size_t i = get_global_id(0);\n"
   "    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"
   "    if(i < outSize)\n"
   "    {\n"
   "		 size_t rowId = in2_row_offsets[i] - indexOffset;\n"
   "    	 size_t row2Id = in2_row_offsets[i+1] - indexOffset;\n"
   "        output[i] = FUNCTIONNAME(&input1[0], &in2_values[rowId], (row2Id-rowId), &in2_col_indices[rowId], const1);\n"
   "        i += gridSize;\n"
   "    }\n"
   "}\n"
);


/*!
 *
 *  OpenCL MapArray kernel for Vector-Matrix. Similar to Map computation but a pointer to first vector is supplied to the user function instead
 *  of just one element.
 */
static std::string MapArrayKernel_CL_Matrix(
   "__kernel void MapArrayKernel_Matrix_KERNELNAME(__global TYPE* input1, __global TYPE* input2, __global TYPE* output, size_t n, size_t xsize, size_t ysize, size_t yoffset, CONST_TYPE const1)\n"
   "{\n"
   "    size_t xindex = get_global_id(0);\n"
   "    size_t yindex = get_global_id(1);\n"
   "    size_t i = yindex*xsize + xindex; \n"
   "    if(i < n && xindex<xsize && yindex <ysize)\n"
   "    {\n"
   "        output[i] = FUNCTIONNAME(&input1[0], input2[i], xindex, yindex+yoffset, const1);\n"
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
 *  \defgroup MapArrayKernels MapArray Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Map skeleton.
 * \{
 */

/*!
 *
 *  CUDA MapArray kernel. Similar to Map computation but a pointer to first vector is supplied to the user function instead
 *  of just one element.
 */
template <typename T, typename ArrayFunc>
__global__ void MapArrayKernel_CU(ArrayFunc mapArrayFunc, T* input1, T* input2, T* output, size_t n)
{
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   while(i < n)
   {
      output[i] = mapArrayFunc.CU(&input1[0], input2[i]);
      i += gridSize;
   }
}




/*!
 *
 *  CUDA MapArray kernel for Vector-Matrix that applies array block-wise. Similar to Map computation but a pointer to first vector is supplied to the user function instead
 *  of just one element.
 */
template <typename T, typename ArrayFunc>
__global__ void MapArrayKernel_CU_Matrix_Blockwise(ArrayFunc mapArrayFunc, T* input1, T* input2, T* output, size_t outSize, size_t p2BlockSize)
{
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   if(i < outSize)
   {
      output[i] = mapArrayFunc.CU(&input1[0], &input2[i*p2BlockSize]);
      i += gridSize;
   }
}


/*!
 *
 *  CUDA MapArray kernel for Vector-SparseMatrix that applies array block-wise. Similar to Map computation but a pointer to first vector is supplied to the user function instead
 *  of just one element.
 */
template <typename T, typename ArrayFunc>
__global__ void MapArrayKernel_CU_Sparse_Matrix_Blockwise(ArrayFunc mapArrayFunc, T* input1, T* in2_values, size_t *in2_row_offsets, size_t *in2_col_indices, T* output, size_t outSize, size_t indexOffset)
{
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   if(i < outSize)    //m_mapArrayFunc->CPU(&input1[0], &values[rowIdx], rowSize, &col_indices[rowIdx]);
   {
      size_t rowId = in2_row_offsets[i] - indexOffset;
      size_t row2Id = in2_row_offsets[i+1] - indexOffset;
      output[i] = mapArrayFunc.CU(&input1[0], &in2_values[rowId], (row2Id-rowId), &in2_col_indices[rowId]);
      i += gridSize;
   }
}



/*!
 *
 *  CUDA MapArray kernel for Vector-Matrix. Similar to Map computation but a pointer to first vector is supplied to the user function instead
 *  of just one element.
 */
template <typename T, typename ArrayFunc>
__global__ void MapArrayKernel_CU_Matrix(ArrayFunc mapArrayFunc, T* input1, T* input2, T* output, size_t n, size_t xsize, size_t ysize, size_t yoffset)
{
   size_t xindex  = blockIdx.x * blockDim.x + threadIdx.x;
   size_t yindex  = blockIdx.y * blockDim.y + threadIdx.y;
   size_t outaddr = yindex*xsize + xindex; //(gridDim.x * blockDim.x) * yindex + xindex;

   if(outaddr < n && xindex<xsize && yindex <ysize)
   {
      output[outaddr] = mapArrayFunc.CU(&input1[0], input2[outaddr], xindex, yindex+yoffset);
   }
}



/*!
 *  \}
 */

}

#endif

#endif

