/*! \file generate_kernels.h
 *  \brief Contains the OpenCL and CUDA kernels for the Generate skeleton.
 */

#ifndef GENERATE_KERNELS_H
#define GENERATE_KERNELS_H

#ifdef SKEPU_OPENCL

#include <string>

namespace skepu
{

/*!
 *  \ingroup kernels
 */

/*!
 *  \defgroup GenerateKernels Generate Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Generate skeleton.
 * \{
 */

/*!
 *
 *  OpenCL Generate kernel for vector.
 */
static std::string GenerateKernel_CL(
   "__kernel void GenerateKernel_KERNELNAME(__global TYPE* output, size_t numElements, size_t indexOffset, CONST_TYPE const1)\n"
   "{\n"
   "    size_t i = get_global_id(0);\n"
   "    size_t gridSize = get_local_size(0)*get_num_groups(0);\n"
   "    while(i < numElements)\n"
   "    {\n"
   "        output[i] = FUNCTIONNAME(i+indexOffset, const1);\n"
   "        i += gridSize;\n"
   "    }\n"
   "}\n"
);


/*!
 *
 *  OpenCL Generate kernel for matrix.
 */
static std::string GenerateKernel_CL_Matrix(
   "__kernel void GenerateKernel_Matrix_KERNELNAME(__global TYPE* output, size_t numElements, size_t xsize, size_t ysize, size_t yoffset, CONST_TYPE const1)\n"
   "{\n"
   "    size_t xindex = get_global_id(0);\n"
   "    size_t yindex = get_global_id(1);\n"
   "    size_t i = yindex*xsize + xindex; \n"
   "    if(i < numElements && xindex<xsize && yindex <ysize)\n"
   "    {\n"
   "        output[i] = FUNCTIONNAME(xindex, yindex+yoffset, const1);\n"
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
 *  \defgroup GenerateKernels Generate Kernels
 *
 *  Definitions of CUDA and OpenCL kernels for the Generate skeleton.
 * \{
 */

/*!
 *
 *  CUDA Generate kernel for vector.
 */
template <typename T, typename GenerateFunc>
__global__ void GenerateKernel_CU(GenerateFunc generateFunc, T* output, size_t numElements, size_t indexOffset)
{
   size_t i = blockIdx.x * blockDim.x + threadIdx.x;
   size_t gridSize = blockDim.x*gridDim.x;

   while(i < numElements)
   {
      output[i] = generateFunc.CU(i+indexOffset);
      i += gridSize;
   }
}



/*!
 *
 *  CUDA Generate kernel for matrix.
 */
template <typename T, typename GenerateFunc>
__global__ void GenerateKernel_CU_Matrix(GenerateFunc generateFunc, T* output, size_t numElements, size_t xsize, size_t ysize, size_t yoffset)
{
   size_t xindex  = blockIdx.x * blockDim.x + threadIdx.x;
   size_t yindex  = blockIdx.y * blockDim.y + threadIdx.y;
   size_t outaddr = yindex*xsize + xindex;

   size_t gridSize = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
   while(outaddr < numElements && xindex<xsize && yindex <ysize)
   {
      output[outaddr] = generateFunc.CU(xindex, yindex+yoffset);
      outaddr += gridSize;
      xindex += blockDim.x*gridDim.x;
      yindex += blockDim.y*gridDim.y;
   }
//    if(outaddr < numElements && xindex<xsize && yindex <ysize)
//    {
//        output[outaddr] = generateFunc.CU(xindex, yindex+yoffset);
//    }



}

/*!
 *  \}
 */

}

#endif

#endif

