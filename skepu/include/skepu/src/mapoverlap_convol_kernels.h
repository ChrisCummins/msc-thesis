/*! \file mapoverlap_convol_kernels.h
 *  \brief Contains the OpenCL and CUDA kernels for the MapOverlap convolution which supports overlap of neighbouring elements.
 */

#ifndef MAPOVERLAP_CONVOL_KERNELS_H
#define MAPOVERLAP_CONVOL_KERNELS_H

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
* The mapoverlap OpenCL kernel with a filter matrix to apply on neighbourhood of each element in the matrix.
*/
static std::string MatrixConvolSharedFilter_CL(
   "__kernel void conv_opencl_shared_filter_KERNELNAME(__global TYPE* input, __global TYPE* output, __constant TYPE* filter, size_t in_rows, size_t in_cols, size_t out_rows, size_t out_cols, size_t filter_rows, size_t filter_cols, size_t in_pitch, size_t out_pitch, size_t sharedRows, size_t sharedCols, __local TYPE* sdata)\n"
   "{\n"
   "    size_t xx = ( (size_t)(get_global_id(0)/get_local_size(0))) * get_local_size(0);\n"
   "    size_t yy = ( (size_t)(get_global_id(1)/get_local_size(1))) * get_local_size(1);\n"
   "    size_t x = get_global_id(0);\n"
   "    size_t y = get_global_id(1);\n"
   "    if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))\n"
   "    {\n"
   "	    size_t sharedIdx = get_local_id(1) * sharedCols + get_local_id(0);\n"
   "	    sdata[sharedIdx]= input[y*in_pitch + x];\n"
   "	    size_t shared_x= get_local_id(0)+get_local_size(0);\n"
   "	    size_t shared_y= get_local_id(1);\n"
   "	    while(shared_y<sharedRows)\n"
   "	    {\n"
   "		    while(shared_x<sharedCols)\n"
   "		    {\n"
   "		    	sharedIdx = shared_y * sharedCols + shared_x; \n"
   "		    	sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];\n"
   "		    	shared_x = shared_x + get_local_size(0);\n"
   "		    }\n"
   "		    shared_x = get_local_id(0);\n"
   "		    shared_y = shared_y + get_local_size(1);\n"
   "	    }	    \n"
   "   }\n"
   "   barrier(CLK_LOCAL_MEM_FENCE);\n"
   "   if(x<out_cols && y<out_rows)\n"
   "   {\n"
   "	    TYPE sum=0;\n"
   "		for(size_t j=0;j<filter_rows;j++) \n"
   "		{\n"
   "			for(size_t i=0;i<filter_cols;i++) \n"
   "			{\n"
   "				sum += sdata[(get_local_id(1)+j) * sharedCols + (get_local_id(0)+i) ] * filter[j*filter_cols+i];\n"
   "			}\n"
   "		}\n"
   "	 	output[y*out_pitch+x] = sum / (filter_rows * filter_cols);\n"
   "   }\n"
   "}"
);



/*!
* The mapoverlap OpenCL kernel to apply a user function on neighbourhood of each element in the matrix.
*/
static std::string MatrixConvol2D_CL(
   "__kernel void conv_opencl_2D_KERNELNAME(__global TYPE* input, __global TYPE* output, size_t out_rows, size_t out_cols, size_t filter_rows, size_t filter_cols, size_t in_pitch, size_t out_pitch, size_t stride, size_t sharedRows, size_t sharedCols, __local TYPE* sdata)\n"
   "{\n"
   "    size_t xx = ( (size_t)(get_global_id(0)/get_local_size(0))) * get_local_size(0);\n"
   "    size_t yy = ( (size_t)(get_global_id(1)/get_local_size(1))) * get_local_size(1);\n"
   "    size_t x = get_global_id(0);\n"
   "    size_t y = get_global_id(1);\n"
   "    if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))\n"
   "    {\n"
   "	    size_t sharedIdx = get_local_id(1) * sharedCols + get_local_id(0);\n"
   "	    sdata[sharedIdx]= input[y*in_pitch + x];\n"
   "	    size_t shared_x= get_local_id(0)+get_local_size(0);\n"
   "	    size_t shared_y= get_local_id(1);\n"
   "	    while(shared_y<sharedRows)\n"
   "	    {\n"
   "		    while(shared_x<sharedCols)\n"
   "		    {\n"
   "		    	sharedIdx = shared_y * sharedCols + shared_x; \n"
   "		    	sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];\n"
   "		    	shared_x = shared_x + get_local_size(0);\n"
   "		    }\n"
   "		    shared_x = get_local_id(0);\n"
   "		    shared_y = shared_y + get_local_size(1);\n"
   "	    }	    \n"
   "   }\n"
   "   barrier(CLK_LOCAL_MEM_FENCE);\n"
   "   if(x<out_cols && y<out_rows)\n"
   "   {\n"
   "       output[y*out_pitch+x] = FUNCTIONNAME(&(sdata[(get_local_id(1)+(filter_rows/2)) * sharedCols + (get_local_id(0)+(filter_cols/2))]), stride);\n"
   "   }\n"
   "}"
);


/*!
* The mapoverlap OpenCL kernel to apply on neighbourhood of each element in the matrix.
*/
static std::string MatrixConvolShared_CL(
   "__kernel void conv_opencl_shared_KERNELNAME(__global TYPE* input, __global TYPE* output, size_t in_rows, size_t in_cols, size_t out_rows, size_t out_cols, size_t filter_rows, size_t filter_cols, size_t in_pitch, size_t out_pitch, size_t sharedRows, size_t sharedCols, __local TYPE* sdata)\n"
   "{\n"
   "    size_t xx = ( (size_t)(get_global_id(0)/get_local_size(0))) * get_local_size(0);\n"
   "    size_t yy = ( (size_t)(get_global_id(1)/get_local_size(1))) * get_local_size(1);\n"
   "    size_t x = get_global_id(0);\n"
   "    size_t y = get_global_id(1);\n"
   "    if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))\n"
   "    {\n"
   "	    size_t sharedIdx = get_local_id(1) * sharedCols + get_local_id(0);\n"
   "	    sdata[sharedIdx]= input[y*in_pitch + x];\n"
   "	    size_t shared_x= get_local_id(0)+get_local_size(0);\n"
   "	    size_t shared_y= get_local_id(1);\n"
   "	    while(shared_y<sharedRows)\n"
   "	    {\n"
   "		    while(shared_x<sharedCols)\n"
   "		    {\n"
   "		    	sharedIdx = shared_y * sharedCols + shared_x; \n"
   "		    	sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];\n"
   "		    	shared_x = shared_x + get_local_size(0);\n"
   "		    }\n"
   "		    shared_x = get_local_id(0);\n"
   "		    shared_y = shared_y + get_local_size(1);\n"
   "	    }	    \n"
   "   }\n"
   "   barrier(CLK_LOCAL_MEM_FENCE);\n"
   "   if(x<out_cols && y<out_rows)\n"
   "   {\n"
   "	    TYPE sum=0;\n"
   "		for(size_t j=0;j<filter_rows;j++) \n"
   "		{\n"
   "			for(size_t i=0;i<filter_cols;i++) \n"
   "			{\n"
   "				sum += sdata[(get_local_id(1)+j) * sharedCols + (get_local_id(0)+i) ];\n"
   "			}\n"
   "		}\n"
   "	 	output[y*out_pitch+x] = sum / (filter_rows * filter_cols);\n"
   "   }\n"
   "}"
);


/*!
 *  \}
 */

}


#endif

//#################
//-----------------
//#################


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


#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 32
#define WARP_SIZE 32
#define NUM_REGISTERS_PER_SP 32768
#define SHARED_MEM_SIZE_BYTES 48000
#define THREADS_PER_WARP 32
#define WARPS_PER_SP 48
#define THREAD_BLOCK_PER_SP 8


/*!
* Helper: to calculate maximum.
*/
template <typename T>
T max(T a, T b)
{
   return (a>b)? a:b;
}

/*!
* Helper: to calculate minimum.
*/
template <typename T>
T min(T a, T b)
{
   return (a<b)? a:b;
}




/*!
* Helper: to calculate tiling factor.
*/
template <typename T>
size_t calculateTiling(size_t regCountPerThread, size_t filterSizeX, size_t filterSizeY, size_t inputSizeX, bool maximizeTiling=false)
{
   size_t numThreadsPerTB = (BLOCK_SIZE_X * BLOCK_SIZE_Y);

   size_t numWarpsPerTB = (numThreadsPerTB+WARP_SIZE-1) / WARP_SIZE;

   size_t maxTBPerSP = min( (WARPS_PER_SP / numWarpsPerTB), (size_t)THREAD_BLOCK_PER_SP);

   if(maximizeTiling)
      maxTBPerSP = 1;
   else
      maxTBPerSP = 2; // limit to 2, not full occupancy


   long long remRegPerThreads = NUM_REGISTERS_PER_SP - (regCountPerThread * numWarpsPerTB * WARP_SIZE * maxTBPerSP); // * maxTBPerSP

   if(remRegPerThreads <0)
   {
      std::cerr << "Error! Limited by Register usage, tiling cannot be more than 1\n";
      return 1;
   }

   remRegPerThreads = remRegPerThreads / (numWarpsPerTB * WARP_SIZE * maxTBPerSP); //maxTBPerSP); // tiling cannot be more than this

   long long sharedMem =  SHARED_MEM_SIZE_BYTES - ((BLOCK_SIZE_X + filterSizeX - 1) * (BLOCK_SIZE_Y + filterSizeY - 1) * sizeof(T) * maxTBPerSP); // * sizeof(T) * maxTBPerSP);

   if(sharedMem < 0)
   {
      std::cerr << "Error! Limited by shared memory usage, tiling cannot be more than 1\n";
      return 1;
   }

   size_t tilingSM = min( (size_t)(inputSizeX/BLOCK_SIZE_X), (size_t)(sharedMem / (BLOCK_SIZE_X * (BLOCK_SIZE_Y + filterSizeY - 1) * sizeof (T) * maxTBPerSP)) ); // * maxTBPerSP);

   tilingSM = min(tilingSM, (size_t)remRegPerThreads); // assuming a tile increase register count by one.

   inputSizeX = inputSizeX / BLOCK_SIZE_X;
   if(tilingSM>1)
   {
      while( (inputSizeX%tilingSM) != 0)
      {
         tilingSM--;
      }
   }
   else
      tilingSM = 1;

   return tilingSM;
}



// constant buffer used to store filter...
__device__ __constant__ char deviceFilter[16386];




/*!
 *  The 2D mapoverlap CUDA kernel to apply the given user function on neighbourhood of each element in the matrix.
 *
 */
template<typename T, typename OverlapFunc>
__global__ void conv_cuda_2D_kernel(OverlapFunc mapOverlapFunc, T* input, T* output, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
   size_t y = yy + threadIdx.y;

   if( x<(out_cols+(filter_cols-1)) && y<(out_rows+(filter_rows-1)) )
   {
      size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;

      sdata[sharedIdx]= input[y*in_pitch + x];

      size_t shared_x= threadIdx.x+blockDim.x;
      size_t shared_y= threadIdx.y;

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }
   __syncthreads();

   if(x<out_cols && y<out_rows)
   {
      output[y*out_pitch+x] = mapOverlapFunc.CU(&(sdata[(threadIdx.y+(filter_rows/2)) * sharedCols + (threadIdx.x+(filter_cols/2))]));
   }
}



/*!
 *  The mapoverlap CUDA kernel with (or without) a filter matrix to apply on neighbourhood of each element in the matrix.
 *
 */
template<bool useFilter, typename T>
__global__ void conv_cuda_shared_kernel(T* input, T* output, const size_t in_rows, const size_t in_cols, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
   size_t y = yy + threadIdx.y;

   if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))
   {
      size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;

      sdata[sharedIdx]= input[y*in_pitch + x];

      size_t shared_x= threadIdx.x+blockDim.x;
      size_t shared_y= threadIdx.y;

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }
   __syncthreads();

   if(x<out_cols && y<out_rows)
   {
      T sum=0;

      if(useFilter)
      {
         T *d_Filter = reinterpret_cast<T*>(deviceFilter);
         for(size_t j=0; j<filter_rows; j++)
         {
            for(size_t i=0; i<filter_cols; i++)
            {
               sum += sdata[(threadIdx.y+j) * sharedCols + (threadIdx.x+i) ] * d_Filter[j*filter_cols+i];
            }
         }
      }
      else
      {
         for(size_t j=0; j<filter_rows; j++)
         {
            for(size_t i=0; i<filter_cols; i++)
            {
               sum += sdata[(threadIdx.y+j) * sharedCols + (threadIdx.x+i) ];
            }
         }
      }
      output[y*out_pitch+x] = sum / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
   }
}



/*!
 *  The mapoverlap CUDA kernel with tiling support; with (or without) a filter
 *  matrix to apply on neighbourhood of each element in the matrix.
 *
 */
template<bool useFilter, typename T>
__global__ void conv_cuda_shared_tiling_kernel(T* input, T* output, const size_t numTiles, const size_t in_cols, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x *  numTiles;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
   size_t y = yy + threadIdx.y;

   size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;
   size_t shared_x= threadIdx.x+blockDim.x;
   size_t shared_y= threadIdx.y;

   if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))
   {
      sdata[sharedIdx]= input[y*in_pitch + x];

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }

   __syncthreads();

   sharedIdx = threadIdx.x;

   for(size_t t=0; t<numTiles; t++)
   {
      if(x<out_cols && y<out_rows)
      {
//		    T sum=0;
         shared_x = 0;

         if(useFilter)
         {
            T *d_Filter = reinterpret_cast<T*>(deviceFilter);
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x += sdata[(threadIdx.y+j) * sharedCols + (sharedIdx+i) ] * d_Filter[j*filter_cols+i];
               }
            }
         }
         else
         {
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x += sdata[(threadIdx.y+j) * sharedCols + (sharedIdx+i) ];
               }
            }
         }
         output[y*out_pitch+x] = shared_x / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         x += blockDim.x;
         sharedIdx += blockDim.x;
      }
   }
}





/*!
 *  The mapoverlap CUDA kernel with tiling support (tiling factor: 2); with (or without) a filter
 *  matrix to apply on neighbourhood of each element in the matrix.
 *
 */
template<bool useFilter, typename T>
__global__ void conv_cuda_shared_tiling_2_kernel(T* input, T* output, const size_t in_cols, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x *  2;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
   size_t y = yy + threadIdx.y;

   size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;


   if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))
   {
      sdata[sharedIdx]= input[y*in_pitch + x];

      size_t shared_x= threadIdx.x+blockDim.x;
      size_t shared_y= threadIdx.y;

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }

   __syncthreads();

   sharedIdx = threadIdx.x;

//	for(size_t t=0;t<numTiles; t++)
   {
      if(x<out_cols && y<out_rows)
      {
         T sum=0;
         T sum2=0;

         if(useFilter)
         {
            T *d_Filter = reinterpret_cast<T*>(deviceFilter);
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  sum += sdata[(threadIdx.y+j) * sharedCols + (sharedIdx+i) ] * d_Filter[j*filter_cols+i];
                  sum2 += sdata[(threadIdx.y+j) * sharedCols + (sharedIdx+blockDim.x+i) ] * d_Filter[j*filter_cols+i];
               }
            }
         }
         else
         {
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  sum += sdata[(threadIdx.y+j) * sharedCols + (sharedIdx+i) ];
                  sum2 += sdata[(threadIdx.y+j) * sharedCols + (sharedIdx+blockDim.x+i) ];
               }
            }
         }
         output[y*out_pitch+x] = sum / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         output[y*out_pitch+x+blockDim.x] = sum2 / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
//		 	x += blockDim.x;
//		 	sharedIdx += blockDim.x;
      }
   }
}


/*!
 *  The mapoverlap CUDA kernel with tiling support (tiling factor: 4); with (or without) a filter
 *  matrix to apply on neighbourhood of each element in the matrix.
 *
 */
template<bool useFilter, typename T>
__global__ void conv_cuda_shared_tiling_4_kernel(T* input, T* output, const size_t in_cols, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x *  4;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
//    size_t x_in = xx  + threadIdx.x;
   size_t y = yy + threadIdx.y;

   size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;

   size_t shared_x= threadIdx.x+blockDim.x;


   if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))
   {
      sdata[sharedIdx]= input[y*in_pitch + x];

      size_t shared_y= threadIdx.y;

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }

   __syncthreads();

   sharedIdx = threadIdx.x;

//	for(size_t t=0;t<numTiles; t++)
   {
      if(x<out_cols && y<out_rows)
      {
         T sum=0;
         T sum2=0;
         T sum3=0;
         T sum4=0;

         if(useFilter)
         {
            T *d_Filter = reinterpret_cast<T*>(deviceFilter);
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x] * d_Filter[j*filter_cols+i];
               }
            }
         }
         else
         {
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x];
               }
            }
         }
         shared_x = y*out_pitch+x;
         output[shared_x] = sum / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum2 / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum3 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum4 / (filter_rows * filter_cols);
      }
   }
}

/*!
 *  The mapoverlap CUDA kernel with tiling support (tiling factor: 6); with (or without) a filter
 *  matrix to apply on neighbourhood of each element in the matrix.
 *
 */
template<bool useFilter, typename T>
__global__ void conv_cuda_shared_tiling_6_kernel(T* input, T* output, const size_t in_cols, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x *  6;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
//    size_t x_in = xx  + threadIdx.x;
   size_t y = yy + threadIdx.y;

   size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;

   size_t shared_x= threadIdx.x+blockDim.x;


   if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))
   {
      sdata[sharedIdx]= input[y*in_pitch + x];

      size_t shared_y= threadIdx.y;

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }

   __syncthreads();

   sharedIdx = threadIdx.x;

//	for(size_t t=0;t<numTiles; t++)
   {
      if(x<out_cols && y<out_rows)
      {
         T sum=0;
         T sum2=0;
         T sum3=0;
         T sum4=0;
         T sum5=0;
         T sum6=0;

         if(useFilter)
         {
            T *d_Filter = reinterpret_cast<T*>(deviceFilter);
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x] * d_Filter[j*filter_cols+i];
               }
            }
         }
         else
         {
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x];
               }
            }
         }
         shared_x = y*out_pitch+x;
         output[shared_x] = sum / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum2 / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum3 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum4 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum5 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum6 / (filter_rows * filter_cols);
      }
   }
}



/*!
 *  The mapoverlap CUDA kernel with tiling support (tiling factor: 8); with (or without) a filter
 *  matrix to apply on neighbourhood of each element in the matrix.
 *
 */
template<bool useFilter, typename T>
__global__ void conv_cuda_shared_tiling_8_kernel(T* input, T* output, const size_t in_cols, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x *  8;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
//    size_t x_in = xx  + threadIdx.x;
   size_t y = yy + threadIdx.y;

   size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;

   size_t shared_x= threadIdx.x+blockDim.x;


   if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))
   {
      sdata[sharedIdx]= input[y*in_pitch + x];

      size_t shared_y= threadIdx.y;

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }

   __syncthreads();

   sharedIdx = threadIdx.x;

//	for(size_t t=0;t<numTiles; t++)
   {
      if(x<out_cols && y<out_rows)
      {
         T sum=0;
         T sum2=0;
         T sum3=0;
         T sum4=0;
         T sum5=0;
         T sum6=0;
         T sum7=0;
         T sum8=0;

         if(useFilter)
         {
            T *d_Filter = reinterpret_cast<T*>(deviceFilter);
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum7 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum8 += sdata[shared_x] * d_Filter[j*filter_cols+i];
               }
            }
         }
         else
         {
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum7 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum8 += sdata[shared_x];
               }
            }
         }
         shared_x = y*out_pitch+x;
         output[shared_x] = sum / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum2 / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum3 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum4 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum5 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum6 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum7 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum8 / (filter_rows * filter_cols);
      }
   }
}




/*!
 *  The mapoverlap CUDA kernel with tiling support (tiling factor: 10); with (or without) a filter
 *  matrix to apply on neighbourhood of each element in the matrix.
 *
 */
template<bool useFilter, typename T>
__global__ void conv_cuda_shared_tiling_10_kernel(T* input, T* output, const size_t in_cols, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x *  10;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
//    size_t x_in = xx  + threadIdx.x;
   size_t y = yy + threadIdx.y;

   size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;

   size_t shared_x= threadIdx.x+blockDim.x;


   if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))
   {
      sdata[sharedIdx]= input[y*in_pitch + x];

      size_t shared_y= threadIdx.y;

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }

   __syncthreads();

   sharedIdx = threadIdx.x;

//	for(size_t t=0;t<numTiles; t++)
   {
      if(x<out_cols && y<out_rows)
      {
         T sum=0;
         T sum2=0;
         T sum3=0;
         T sum4=0;
         T sum5=0;
         T sum6=0;
         T sum7=0;
         T sum8=0;
         T sum9=0;
         T sum10=0;

         if(useFilter)
         {
            T *d_Filter = reinterpret_cast<T*>(deviceFilter);
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum7 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum8 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum9 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum10 += sdata[shared_x] * d_Filter[j*filter_cols+i];
               }
            }
         }
         else
         {
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum7 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum8 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum9 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum10 += sdata[shared_x];
               }
            }
         }
         shared_x = y*out_pitch+x;
         output[shared_x] = sum / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum2 / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum3 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum4 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum5 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum6 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum7 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum8 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum9 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum10 / (filter_rows * filter_cols);
      }
   }
}



/*!
 *  The mapoverlap CUDA kernel with tiling support (tiling factor: 22); with (or without) a filter
 *  matrix to apply on neighbourhood of each element in the matrix.
 *
 */
template<bool useFilter, typename T>
__global__ void conv_cuda_shared_tiling_12_kernel(T* input, T* output, const size_t in_cols, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x *  12;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
//    size_t x_in = xx  + threadIdx.x;
   size_t y = yy + threadIdx.y;

   size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;

   size_t shared_x= threadIdx.x+blockDim.x;


   if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))
   {
      sdata[sharedIdx]= input[y*in_pitch + x];

      size_t shared_y= threadIdx.y;

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }

   __syncthreads();

   sharedIdx = threadIdx.x;

//	for(size_t t=0;t<numTiles; t++)
   {
      if(x<out_cols && y<out_rows)
      {
         T sum=0;
         T sum2=0;
         T sum3=0;
         T sum4=0;
         T sum5=0;
         T sum6=0;
         T sum7=0;
         T sum8=0;
         T sum9=0;
         T sum10=0;
         T sum11=0;
         T sum12=0;

         if(useFilter)
         {
            T *d_Filter = reinterpret_cast<T*>(deviceFilter);
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum7 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum8 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum9 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum10 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum11 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum12 += sdata[shared_x] * d_Filter[j*filter_cols+i];
               }
            }
         }
         else
         {
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum7 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum8 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum9 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum10 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum11 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum12 += sdata[shared_x];
               }
            }
         }
         shared_x = y*out_pitch+x;
         output[shared_x] = sum / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum2 / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum3 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum4 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum5 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum6 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum7 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum8 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum9 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum10 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum11 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum12 / (filter_rows * filter_cols);
      }
   }
}



/*!
 *  The mapoverlap CUDA kernel with tiling support (tiling factor: 14); with (or without) a filter
 *  matrix to apply on neighbourhood of each element in the matrix.
 *
 */
template<bool useFilter, typename T>
__global__ void conv_cuda_shared_tiling_14_kernel(T* input, T* output, const size_t in_cols, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x *  14;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
//    size_t x_in = xx  + threadIdx.x;
   size_t y = yy + threadIdx.y;

   size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;

   size_t shared_x= threadIdx.x+blockDim.x;


   if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))
   {
      sdata[sharedIdx]= input[y*in_pitch + x];

      size_t shared_y= threadIdx.y;

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }

   __syncthreads();

   sharedIdx = threadIdx.x;

//	for(size_t t=0;t<numTiles; t++)
   {
      if(x<out_cols && y<out_rows)
      {
         T sum=0;
         T sum2=0;
         T sum3=0;
         T sum4=0;
         T sum5=0;
         T sum6=0;
         T sum7=0;
         T sum8=0;
         T sum9=0;
         T sum10=0;
         T sum11=0;
         T sum12=0;
         T sum13=0;
         T sum14=0;

         if(useFilter)
         {
            T *d_Filter = reinterpret_cast<T*>(deviceFilter);
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum7 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum8 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum9 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum10 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum11 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum12 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum13 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum14 += sdata[shared_x] * d_Filter[j*filter_cols+i];
               }
            }
         }
         else
         {
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum7 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum8 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum9 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum10 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum11 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum12 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum13 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum14 += sdata[shared_x];
               }
            }
         }
         shared_x = y*out_pitch+x;
         output[shared_x] = sum / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum2 / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum3 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum4 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum5 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum6 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum7 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum8 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum9 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum10 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum11 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum12 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum13 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum14 / (filter_rows * filter_cols);
      }
   }
}




/*!
 *  The mapoverlap CUDA kernel with tiling support (tiling factor: 16); with (or without) a filter
 *  matrix to apply on neighbourhood of each element in the matrix.
 *
 */
template<bool useFilter, typename T>
__global__ void conv_cuda_shared_tiling_16_kernel(T* input, T* output, const size_t in_cols, const size_t out_rows, const size_t out_cols, const size_t filter_rows, const size_t filter_cols, size_t in_pitch, size_t out_pitch, const size_t sharedRows, const size_t sharedCols)
{
   extern __shared__ char _sdata[];
   T* sdata = reinterpret_cast<T*>(_sdata); // will also contain extra (overlap data)

   size_t xx = blockIdx.x * blockDim.x *  16;
   size_t yy = blockIdx.y * blockDim.y;

   size_t x = xx + threadIdx.x;
//    size_t x_in = xx  + threadIdx.x;
   size_t y = yy + threadIdx.y;

   size_t sharedIdx = threadIdx.y * sharedCols + threadIdx.x;

   size_t shared_x= threadIdx.x+blockDim.x;


   if(x<(out_cols+filter_cols-1) && y<(out_rows+filter_rows-1))
   {
      sdata[sharedIdx]= input[y*in_pitch + x];

      size_t shared_y= threadIdx.y;

      // To load data in shared memory including neighbouring elements...
      while(shared_y<sharedRows)
      {
         while(shared_x<sharedCols)
         {
            sharedIdx = shared_y * sharedCols + shared_x;
            sdata[sharedIdx]= input[(yy+shared_y) * in_pitch + xx + shared_x];
            shared_x = shared_x + blockDim.x;
         }
         shared_x = threadIdx.x;
         shared_y = shared_y + blockDim.y;
      }
   }

   __syncthreads();

   sharedIdx = threadIdx.x;

//	for(size_t t=0;t<numTiles; t++)
   {
      if(x<out_cols && y<out_rows)
      {
         T sum=0;
         T sum2=0;
         T sum3=0;
         T sum4=0;
         T sum5=0;
         T sum6=0;
         T sum7=0;
         T sum8=0;
         T sum9=0;
         T sum10=0;
         T sum11=0;
         T sum12=0;
         T sum13=0;
         T sum14=0;
         T sum15=0;
         T sum16=0;

         if(useFilter)
         {
            T *d_Filter = reinterpret_cast<T*>(deviceFilter);
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum7 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum8 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum9 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum10 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum11 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum12 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum13 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum14 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum15 += sdata[shared_x] * d_Filter[j*filter_cols+i];
                  shared_x +=  blockDim.x;
                  sum16 += sdata[shared_x] * d_Filter[j*filter_cols+i];
               }
            }
         }
         else
         {
            for(size_t j=0; j<filter_rows; j++) // 7
            {
               for(size_t i=0; i<filter_cols; i++) // 7
               {
                  shared_x = (threadIdx.y+j) * sharedCols + (sharedIdx+i);
                  sum += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum2 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum3 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum4 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum5 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum6 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum7 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum8 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum9 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum10 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum11 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum12 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum13 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum14 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum15 += sdata[shared_x];
                  shared_x +=  blockDim.x;
                  sum16 += sdata[shared_x];
               }
            }
         }
         shared_x = y*out_pitch+x;
         output[shared_x] = sum / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum2 / (filter_rows * filter_cols); //sdata[(threadIdx.y+2) * sharedCols + (threadIdx.x+2) ];
         shared_x +=  blockDim.x;
         output[shared_x] = sum3 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum4 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum5 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum6 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum7 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum8 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum9 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum10 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum11 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum12 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum13 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum14 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum15 / (filter_rows * filter_cols);
         shared_x +=  blockDim.x;
         output[shared_x] = sum16 / (filter_rows * filter_cols);
      }
   }
}


/*!
 *  \}
 */

} // end namespace skepu

#endif


#endif

