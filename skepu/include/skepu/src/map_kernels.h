/*! \file map_kernels.h
 *  \brief Contains the OpenCL and CUDA kernels for the Map skeleton.
 */

#ifndef MAP_KERNELS_H
#define MAP_KERNELS_H

#ifdef SKEPU_OPENCL

#include <string>

namespace skepu
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup MapKernels Map Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Map skeleton.
 * \{
 */

/*!
 *
 *  OpenCL Map kernel for \em unary user functions.
 */
static std::string UnaryMapKernel_CL(
   "__kernel void UnaryMapKernel_KERNELNAME(__global TYPE* input, __global TYPE* output, size_t numElements, CONST_TYPE const1)\n"
   "{\n"
   "    int i = get_global_id(0);\n"
   "    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"
   "    while(i < numElements)\n"
   "    {\n"
   "        output[i] = FUNCTIONNAME(input[i], const1);\n"
   "        i += gridSize;\n"
   "    }\n"
   "}\n"
);

/*!
 *
 *  OpenCL Map kernel for \em binary user functions.
 */
static std::string BinaryMapKernel_CL(
   "__kernel void BinaryMapKernel_KERNELNAME(__global TYPE* input1, __global TYPE* input2, __global TYPE* output, size_t n, CONST_TYPE const1)\n"
   "{\n"
   "    int i = get_global_id(0);\n"
   "    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"
   "    while(i < n)\n"
   "    {\n"
   "        output[i] = FUNCTIONNAME(input1[i], input2[i], const1);\n"
   "        i += gridSize;\n"
   "    }\n"
   "}\n"
);

/*!
 *
 *  OpenCL Map kernel for \em trinary user functions.
 */
static std::string TrinaryMapKernel_CL(
   "__kernel void TrinaryMapKernel_KERNELNAME(__global TYPE* input1, __global TYPE* input2, __global TYPE* input3, __global TYPE* output, size_t n, CONST_TYPE const1)\n"
   "{\n"
   "    int i = get_global_id(0);\n"
   "    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"
   "    while(i < n)\n"
   "    {\n"
   "        output[i] = FUNCTIONNAME(input1[i], input2[i], input3[i], const1);\n"
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
 *  \defgroup MapKernels Map Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Map skeleton.
 * \{
 */

/*!
 *
 *  CUDA Map kernel for \em unary user functions.
 */
template <typename T, typename UnaryFunc>
__global__ void MapKernelUnary_CU(UnaryFunc mapFunc, T* input, T* output, size_t n)
{
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   while(i < n)
   {
      output[i] = mapFunc.CU(input[i]);
      i += gridSize;
   }
}

/*!
 *
 *  CUDA Map kernel for \em binary user functions.
 */
template <typename T, typename BinaryFunc>
__global__ void MapKernelBinary_CU(BinaryFunc mapFunc, T* input1, T* input2, T* output, size_t n)
{
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   while(i < n)
   {
      output[i] = mapFunc.CU(input1[i], input2[i]);
      i += gridSize;
   }
}

/*!
 *
 *  CUDA Map kernel for \em trinary user functions.
 */
template <typename T, typename TrinaryFunc>
__global__ void MapKernelTrinary_CU(TrinaryFunc mapFunc, T* input1, T* input2, T* input3, T* output, size_t n)
{
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   while(i < n)
   {
      output[i] = mapFunc.CU(input1[i], input2[i], input3[i]);
      i += gridSize;
   }
}

/*!
 *  \}
 */

}

#endif

#endif

