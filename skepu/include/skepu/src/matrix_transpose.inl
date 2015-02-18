/*! \file matrix_transpose.inl
 *  \brief Contains the definitions of the Matrix transpose functions for CPU, CUDA, OpenMP and OpenCL.
 */

#ifdef SKEPU_OPENMP
#include <omp.h>
#endif

namespace skepu
{

/*!
 * \brief A method to take Matrix transpose on \em CPU backend.
 */
template <typename T>
void Matrix<T>::transpose_CPU()
{
   DEBUG_TEXT_LEVEL1("TRANSPOSE CPU\n")

   if(m_transpose_matrix && m_transpose_matrix->m_rows==m_cols && m_transpose_matrix->m_cols==m_rows && !m_dataChanged)
      return;

   updateHost();

   if(!m_transpose_matrix) // if not created alreay, create transpose matrix
   {
      m_transpose_matrix = new Matrix<T>();
      m_transpose_matrix->resize(m_cols,m_rows);
   }
   else if (m_transpose_matrix->m_rows!=m_cols || m_transpose_matrix->m_cols!=m_rows)
      m_transpose_matrix->resize(m_cols, m_rows); // resize it
   else
      m_transpose_matrix->invalidateDeviceData(); // invalidate any device copies

   for (size_t i=0; i < m_rows; i++)
      for (size_t j=0; j < m_cols; j++)
      {
         m_transpose_matrix->m_data[j*m_rows+i] = m_data[i*m_cols+j];
      }
}

/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/************************                        TRANSPOSE OPENMP                            *******************************/
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

#ifdef SKEPU_OPENMP

/*!
 * \brief A method to take Matrix transpose on \em OpenMP backend.
 */
template <typename T>
void Matrix<T>::transpose_OMP()
{
   DEBUG_TEXT_LEVEL1("TRANSPOSE OPENMP\n")

   if(m_transpose_matrix && m_transpose_matrix->m_rows==m_cols && m_transpose_matrix->m_cols==m_rows && !m_dataChanged)
      return;

   updateHost();

   if(!m_transpose_matrix) // if not created alreay, create transpose matrix
   {
      m_transpose_matrix = new Matrix<T>();
      m_transpose_matrix->resize(m_cols,m_rows);
   }
   else if (m_transpose_matrix->m_rows!=m_cols || m_transpose_matrix->m_cols!=m_rows)
      m_transpose_matrix->resize(m_cols, m_rows); // resize it
   else
      m_transpose_matrix->invalidateDeviceData(); // invalidate any device copies

   #pragma omp parallel for
   for (size_t i=0; i < m_rows; i++)
      for (size_t j=0; j < m_cols; j++)
      {
         m_transpose_matrix->m_data[j*m_rows+i] = m_data[i*m_cols+j];
      }
}

#endif




/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/************************                        TRANSPOSE CUDA                              *******************************/
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



#ifdef SKEPU_CUDA


// ATTENTION: DONT change any of these parameters.
#define TILE_DIM    16
#define BLOCK_ROWS  16


// -------------------------------------------------------
// Transposes Kernels
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------
/*!
 * \brief A näive CUDA kernel to take Matrix transpose.
 */
template <typename T>
__global__ void transposeNaive(T *odata, T* idata, size_t width, size_t height)
{
//	size_t size = width *height;
   size_t xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
   size_t yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

   size_t index_in  = xIndex + width * yIndex;
   size_t index_out = yIndex + height * xIndex;

//  for (size_t i=0; i<TILE_DIM; i+=BLOCK_ROWS)
//  {
   if(xIndex<width && yIndex<height)
      odata[index_out] = idata[index_in];
//    	odata[index_out+i] = idata[index_in+i*width];
//  }
}


/*!
 * \brief An optimized CUDA kernel to take Matrix transpose.
 */
template <typename T>
__global__ void transposeNoBankConflicts(T *odata, T *idata, size_t width, size_t height)
{
   __shared__ T tile[TILE_DIM][TILE_DIM+1];

   size_t xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
   size_t yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
   size_t index_in = xIndex + (yIndex)*width;



//   for (size_t i=0; i<TILE_DIM; i+=BLOCK_ROWS)
//   {
   if(xIndex<width && yIndex<height)
      tile[threadIdx.y][threadIdx.x] = idata[index_in];
//      	tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
//   }

   xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
   yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
   size_t index_out = xIndex + (yIndex)*height;


   __syncthreads();

//    for (size_t i=0; i<TILE_DIM; i+=BLOCK_ROWS)
//    {
   if(xIndex<height && yIndex<width)
      odata[index_out] = tile[threadIdx.x][threadIdx.y];
//      	odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
//    }
}


/*!
 * \brief A method to take Matrix transpose on \em CUDA backend. Always uses 1 CUDA GPU for transpose even if multiple GPUs are available.
 */
template <typename T>
void Matrix<T>::transpose_CU(Device_CU *device)
{
   DEBUG_TEXT_LEVEL1("TRANSPOSE CUDA\n")

   unsigned int deviceID = device->getDeviceID();

   if(m_transpose_matrix && m_transpose_matrix->m_rows==m_cols && m_transpose_matrix->m_cols==m_rows && !m_dataChanged)
      return;

   if(!m_transpose_matrix) // if not created alreay, create transpose matrix
   {
      m_transpose_matrix = new Matrix<T>();
      m_transpose_matrix->resize(m_cols,m_rows);
   }
   else if (m_transpose_matrix->m_rows!=m_cols || m_transpose_matrix->m_cols!=m_rows)
      m_transpose_matrix->resize(m_cols, m_rows); // resize it
   else
      m_transpose_matrix->invalidateDeviceData(); // invalidate any device copies, could optimize it for CUDA as ask for validating copies except this one.

   cudaSetDevice(deviceID);

   typename Matrix<T>::device_pointer_type_cu in_mem_p = updateDevice_CU(&m_data[0], m_rows*m_cols, deviceID, true, false);
   typename Matrix<T>::device_pointer_type_cu out_mem_p = m_transpose_matrix->updateDevice_CU(&(m_transpose_matrix->m_data[0]), m_cols*m_rows, deviceID, false, true);

   // execution configuration parameters
   dim3 grid((m_cols+TILE_DIM-1)/TILE_DIM, (m_rows+TILE_DIM-1)/TILE_DIM), threads(TILE_DIM,BLOCK_ROWS);

#ifdef USE_PINNED_MEMORY
   transposeNoBankConflicts<<<grid, threads,0, (device->m_streams[0])>>>(out_mem_p->getDeviceDataPointer(), in_mem_p->getDeviceDataPointer(), m_cols, m_rows);
#else
   transposeNoBankConflicts<<<grid, threads>>>(out_mem_p->getDeviceDataPointer(), in_mem_p->getDeviceDataPointer(), m_cols, m_rows);
#endif

   out_mem_p->changeDeviceData();
}

#endif





/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/************************                        TRANSPOSE OPENCL                            *******************************/
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



#ifdef SKEPU_OPENCL

/*!
 * \brief A method to take Matrix transpose on \em OpenCL backend. Always uses 1 OpenCL device for transpose even if multiple OpenCL devices are available.
 */
template <typename T>
void Matrix<T>::transpose_CL(unsigned int deviceID)
{
   DEBUG_TEXT_LEVEL1("TRANSPOSE OpenCL\n")

   if(m_transpose_matrix && m_transpose_matrix->m_rows==m_cols && m_transpose_matrix->m_cols==m_rows && !m_dataChanged)
      return;

   if(!m_transpose_matrix) // if not created alreay, create transpose matrix
   {
      m_transpose_matrix = new Matrix<T>();
      m_transpose_matrix->resize(m_cols,m_rows);
   }
   else if (m_transpose_matrix->m_rows!=m_cols || m_transpose_matrix->m_cols!=m_rows)
      m_transpose_matrix->resize(m_cols, m_rows); // resize it
   else
      m_transpose_matrix->invalidateDeviceData(); // invalidate any device copies, could optimize it for CUDA as ask for validating copies except this one.

   Device_CL *device = m_transposeKernels_CL->at(deviceID).second;
   cl_kernel kernel = m_transposeKernels_CL->at(deviceID).first;

   typename Matrix<T>::device_pointer_type_cl in_mem_p = updateDevice_CL(&m_data[0], m_rows*m_cols, device, true);
   typename Matrix<T>::device_pointer_type_cl out_mem_p = m_transpose_matrix->updateDevice_CL(&(m_transpose_matrix->m_data[0]), m_cols*m_rows, device, false);

   cl_mem in_p = in_mem_p->getDeviceDataPointer();
   cl_mem out_p = out_mem_p->getDeviceDataPointer();

   size_t globalWorkSize[3];
   size_t localWorkSize[3];

   localWorkSize[0] = TILE_DIM; //(m_cols>TILE_DIM)? TILE_DIM: m_cols;
   localWorkSize[1] = BLOCK_ROWS; //(m_rows>BLOCK_ROWS)? BLOCK_ROWS: m_rows;
   localWorkSize[2] = 1;

   globalWorkSize[0] = ((m_cols+TILE_DIM-1)/TILE_DIM)*localWorkSize[0];
   globalWorkSize[1] = ((m_rows+TILE_DIM-1)/TILE_DIM)*localWorkSize[1];
   globalWorkSize[2] = 1;

   size_t sharedMemSize = sizeof(T) * (TILE_DIM*TILE_DIM);

   size_t rows= m_rows;
   size_t cols = m_cols;

   // Sets the kernel arguments
   clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&out_p);
   clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in_p);
   clSetKernelArg(kernel, 2, sizeof(int), (void*)&cols);
   clSetKernelArg(kernel, 3, sizeof(int), (void*)&rows);
   clSetKernelArg(kernel, 4, sharedMemSize, NULL);

   cl_int err;

   // Launches the kernel (asynchronous)
   err = clEnqueueNDRangeKernel(device->getQueue(), kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if(err != CL_SUCCESS)
   {
      std::cerr<<"Error launching kernel!!\n";
   }

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
   out_mem_p->copyDeviceToHost();
}

#endif


}

