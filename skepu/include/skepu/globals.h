#ifndef _GLOBAL_H_
#define _GLOBAL_H_

/*! \file globals.h
 *  \brief Contains some helper routines and typedefs that are shared by all classes.
 */

#include <string>
#include <iostream>

namespace skepu
{

/* timing is in µs per byte (i.e. slowness, inverse of bandwidth) */
typedef struct
{
   int cpu_id;
   double timing_htd;
   double latency_htd;
   double timing_dth;
   double latency_dth;
   double timing_dtd;
   double latency_dtd;
} DevTimingStruct;

#define MAX_EXEC_PLANS 10

#define MAX_PARAMS 10

#ifndef MAX_DEPTH
#define MAX_DEPTH 10
#endif

#define OVERSAMPLE false

#define MAX_PARAMS_MAP 3

/*!
 * The number of maxmimum devices that can be used in a system.
 * If more devices found, will need to change this number.
 */
#ifndef MAX_GPU_DEVICES
#define MAX_GPU_DEVICES 4
#endif

// based on CUDA programming guide v4.2
#define MAX_POSSIBLE_CUDA_STREAMS_PER_GPU 16

#ifndef SKEPU_NUMGPU
#define SKEPU_NUMGPU 1
#elif defined(SKEPU_CUDA) // Only when CUDA is enabled and multi-GPU is possible
//	#define USE_PINNED_MEMORY
#endif

// used when checking shared memory and overlap exceeding that in mapoverlap_cu.inl
#define SHMEM_SAFITY_BUFFER 30

// use it to specify threshold when reduction operation should shift to CPU instead of continuing it on GPU, used in 2D reduction specially.
#define REDUCE_GPU_THRESHOLD 50

// OpenCL additional code that can be supplied in a file. useful when using a custom type with skeletons
#define OPENCL_SOURCE_FILE_NAME "opencl_datatype_src.cl"


#ifndef SKEPU_CUDA_DEV_ID
#define SKEPU_CUDA_DEV_ID 0
#endif


}


#include "src/debug.h"

#ifdef SKEPU_CUDA
#include "src/skepu_cuda_helpers.h"
#endif

#ifdef SKEPU_OPENCL
#include "src/skepu_opencl_helpers.h"
#endif

#include "src/helper_methods.h"





#endif
