/*! \file operator_type.h
 *  \brief Declares an enumeration with the different user function types.
 */

#ifndef OPERATOR_TYPE_H
#define OPERATOR_TYPE_H

#ifdef SKEPU_OPENCL
#ifdef USE_MAC_OPENCL
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <string>
#endif

#ifdef SKEPU_CUDA
#include <cuda.h>
#endif

namespace skepu
{

enum FuncType
{
   UNARY,
   BINARY,
   TERNARY,
   OVERLAP,
   OVERLAP_2D,
   ARRAY,
   ARRAY_INDEX,
   ARRAY_INDEX_BLOCK_WISE,
   ARRAY_INDEX_SPARSE_BLOCK_WISE,
   GENERATE,
   GENERATE_MATRIX
};

}

#endif

