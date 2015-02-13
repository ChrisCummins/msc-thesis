/*! \file operator_macros.h
 *  \brief Includes the macro files needed for the defined backends.
 */

#ifndef OPERATOR_MACROS_H
#define OPERATOR_MACROS_H

#include "operator_type.h"


// Can use either CUDA or OpenCL at a time, i.e. cannot use both at the same time.

#if defined(SKEPU_OPENCL) && !defined(SKEPU_CUDA)

#include "operator_macros_cl.inl"

#elif !defined(SKEPU_OPENCL) && defined(SKEPU_CUDA)

#include "operator_macros_cu.inl"

#elif !defined(SKEPU_OPENCL) && !defined(SKEPU_CUDA)

#include "operator_macros_cpu.inl"

#else

#error "Error: Cannot use both CUDA and OpenCL at the same time."

#endif

#endif

