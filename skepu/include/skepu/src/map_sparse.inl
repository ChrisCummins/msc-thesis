/*! \file map_sparse.inl
 *  \brief Contains the definitions of map methods for sparse matrices. All (CPU, OpenMP, CUDA,OpenCL) in one file.
 */

#include "map_kernels.h"
#include "device_mem_pointer_cu.h"
#include "device_cu.h"

namespace skepu
{

/*!
 *  Performs the Map on one sparse matrix. With itself as output. The Map skeleton needs to
 *  be created with a unary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A sparse matrix which the mapping will be performed on. It will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(SparseMatrix<T>& input)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use UNARY_FUNC or UNARY_FUNC_CONSTANT macro\n");
   }

#ifdef USE_MULTI_EXEC_PLAN
// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   /*! It might be the case that in a complex program suchg as SPH, some skeleton obj use default exec plan as are not perf relevant... */
   if(m_execPlanMulti != NULL)
   {
      if(input.isSparseMatrixOnDevice_CU(Environment<int>::getInstance()->bestCUDADevID))
      {
         if(input.isModified_CU(cudaDeviceID))
            m_execPlan = &m_execPlanMulti[2];
         else
            m_execPlan = &m_execPlanMulti[1];
      }
      else
         m_execPlan = &m_execPlanMulti[0];
   }
   if (m_execPlan == NULL || m_execPlan->calibrated == false)
      m_execPlan = &m_defPlan;
#endif
#endif
   assert(m_execPlan != NULL && m_execPlan->calibrated);

   size_t size = input.total_nnz();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, 0);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, 0);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input);
#endif
   case CPU_BACKEND:
      return CPU(input);

   default:
      return CPU(input);
   }
}



/*!
 *  Performs the Map on one sparse matrix. With a second sparse matrix as output. The Map skeleton needs to
 *  be created with a unary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A sparse matrix which the mapping will be performed on.
 *  \param output The result Sparse Matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(SparseMatrix<T>& input, SparseMatrix<T>& output)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use UNARY_FUNC or UNARY_FUNC_CONSTANT macro\n");
   }
   if(input.total_nnz() != output.total_nnz())
   {
      SKEPU_WARNING("Input and output sparse matrices are not of equal size. Output sparse matrix will be resized\n");
      output.resize(input, false);
   }

#ifdef USE_MULTI_EXEC_PLAN
// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   /*! It might be the case that in a complex program suchg as SPH, some skeleton obj use default exec plan as are not perf relevant... */
   if(m_execPlanMulti != NULL)
   {
      if(input.isSparseMatrixOnDevice_CU(Environment<int>::getInstance()->bestCUDADevID))
      {
         if(input.isModified_CU(cudaDeviceID))
            m_execPlan = &m_execPlanMulti[2];
         else
            m_execPlan = &m_execPlanMulti[1];
      }
      else
         m_execPlan = &m_execPlanMulti[0];
   }
   if (m_execPlan == NULL || m_execPlan->calibrated == false)
      m_execPlan = &m_defPlan;
#endif
#endif
   assert(m_execPlan != NULL && m_execPlan->calibrated);

   size_t size = input.total_nnz();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output, 0);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, 0);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, output);
#endif
   case CPU_BACKEND:
      return CPU(input, output);

   default:
      return CPU(input, output);
   }
}


/*!
 *  Performs the Map on two sparse matrices. With a seperate output. The Map skeleton needs to
 *  be created with a binary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A sparse matrix which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& output)
{
   if(m_mapFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use BINARY_FUNC or BINARY_FUNC_CONSTANT macro\n");
   }
   
   if(input1.total_nnz()!=input2.total_nnz())
   {
      SKEPU_ERROR("Error! Both Input sparse matrices should be of equal size. Operation aborted\n");
   }

   if(input1.total_nnz() != output.total_nnz())
   {
      SKEPU_ERROR("Input and output sparse matrices are not of equal size. Output sparse matrix will be resized\n");
      output.resize(input1, false);
   }

#ifdef USE_MULTI_EXEC_PLAN
// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   /*! It might be the case that in a complex program suchg as SPH, some skeleton obj use default exec plan as are not perf relevant... */
   if(m_execPlanMulti != NULL)
   {
      /*! both are on GPU... */
      if(input1.isSparseMatrixOnDevice_CU(cudaDeviceID) && input2.isSparseMatrixOnDevice_CU(cudaDeviceID))
      {
         // both are modified on GPU so need to copy back...
         if(input1.isModified_CU(cudaDeviceID) && input2.isModified_CU(cudaDeviceID))
            m_execPlan = &m_execPlanMulti[5];

         // one of them is modified on GPU so need to copy back...
         else if(input1.isModified_CU(cudaDeviceID) || input2.isModified_CU(cudaDeviceID))
            m_execPlan = &m_execPlanMulti[4];

         // no operand is modified on GPU so no need to copy back...
         else
            m_execPlan = &m_execPlanMulti[3];
      }

      /*! one of them is on GPU... */
      else if(input1.isSparseMatrixOnDevice_CU(cudaDeviceID) || input2.isSparseMatrixOnDevice_CU(cudaDeviceID))
      {
         // one that is present on GPU is modified so need to copy back...
         if(input1.isModified_CU(cudaDeviceID) || input2.isModified_CU(cudaDeviceID))
            m_execPlan = &m_execPlanMulti[2];

         // no operand is modified on GPU so no need to copy back...
         else
            m_execPlan = &m_execPlanMulti[1];
      }

      /*! nothing is on GPU... */
      else
         m_execPlan = &m_execPlanMulti[0];
   }
   if (m_execPlan == NULL || m_execPlan->calibrated == false)
      m_execPlan = &m_defPlan;
#endif
#endif
   assert(m_execPlan != NULL && m_execPlan->calibrated);

   size_t size = input1.total_nnz();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, output, 0);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, output, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, output, 0);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, output, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input1, input2, output);
#endif
   case CPU_BACKEND:
      return CPU(input1, input2, output);

   default:
      return CPU(input1, input2, output);
   }
}



/*!
 *  Performs the Map on three sparse matrices. With a seperate output. The Map skeleton needs to
 *  be created with a trinary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A sparse matrix which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param input3 A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& input3, SparseMatrix<T>& output)
{
   if(m_mapFunc->funcType != TERNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use TERNARY_FUNC or TERNARY_FUNC_CONSTANT macro\n");
   }
   
   if(input1.total_nnz()!=input2.total_nnz() || input1.total_nnz()!=input3.total_nnz())
   {
      SKEPU_ERROR("All Input sparse matrices should be of equal size. Operation aborted\n");
   }

   if(input1.total_nnz() != output.total_nnz())
   {
      SKEPU_WARNING("Input and output sparse matrices are not of equal size. Output sparse matrix will be resized\n");
      output.resize(input1, false);
   }
   

#ifdef USE_MULTI_EXEC_PLAN
// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   /*! It might be the case that in a complex program suchg as SPH, some skeleton obj use default exec plan as are not perf relevant... */
   if(m_execPlanMulti != NULL)
   {
      /*! all three are on GPU... */
      if(input1.isSparseMatrixOnDevice_CU(cudaDeviceID) && input2.isSparseMatrixOnDevice_CU(cudaDeviceID) && input3.isSparseMatrixOnDevice_CU(cudaDeviceID))
      {
         // all three are modified on GPU so need to copy back...
         if(input1.isModified_CU(cudaDeviceID) && input2.isModified_CU(cudaDeviceID) && input3.isModified_CU(cudaDeviceID))
            m_execPlan = &m_execPlanMulti[9];

         // two of them are modified on GPU so need to copy back...
         else if( (input1.isModified_CU(cudaDeviceID) && input2.isModified_CU(cudaDeviceID)) ||
                  (input1.isModified_CU(cudaDeviceID) && input3.isModified_CU(cudaDeviceID)) ||
                  (input2.isModified_CU(cudaDeviceID) && input3.isModified_CU(cudaDeviceID)) )
            m_execPlan = &m_execPlanMulti[8];

         // one of them is modified on GPU so need to copy back...
         else if(input1.isModified_CU(cudaDeviceID) || input2.isModified_CU(cudaDeviceID) || input3.isModified_CU(cudaDeviceID))
            m_execPlan = &m_execPlanMulti[7];

         // no operand is modified on GPU so no need to copy back...
         else
            m_execPlan = &m_execPlanMulti[6];
      }

      /*! two of them are present on GPU... */
      else if( (input1.isSparseMatrixOnDevice_CU(cudaDeviceID) && input2.isSparseMatrixOnDevice_CU(cudaDeviceID)) ||
               (input1.isSparseMatrixOnDevice_CU(cudaDeviceID) && input3.isSparseMatrixOnDevice_CU(cudaDeviceID)) ||
               (input2.isSparseMatrixOnDevice_CU(cudaDeviceID) && input3.isSparseMatrixOnDevice_CU(cudaDeviceID)) )
      {
         // both that are present on GPU are modified...
         if( (input1.isModified_CU(cudaDeviceID) && input2.isModified_CU(cudaDeviceID)) ||
               (input1.isModified_CU(cudaDeviceID) && input3.isModified_CU(cudaDeviceID)) ||
               (input2.isModified_CU(cudaDeviceID) && input3.isModified_CU(cudaDeviceID)) )
            m_execPlan = &m_execPlanMulti[5];

         // one of them is modified on GPU so need to copy back...
         else if(input1.isModified_CU(cudaDeviceID) || input2.isModified_CU(cudaDeviceID) || input3.isModified_CU(cudaDeviceID))
            m_execPlan = &m_execPlanMulti[4];

         // no operand is modified on GPU so no need to copy back...
         else
            m_execPlan = &m_execPlanMulti[3];
      }

      /*! one of them is present on GPU... */
      else if (input1.isSparseMatrixOnDevice_CU(cudaDeviceID) || input2.isSparseMatrixOnDevice_CU(cudaDeviceID) || input3.isSparseMatrixOnDevice_CU(cudaDeviceID))
      {
         // one that is present is also modified on GPU so need to copy back...
         if(input1.isModified_CU(cudaDeviceID) || input2.isModified_CU(cudaDeviceID) || input3.isModified_CU(cudaDeviceID))
            m_execPlan = &m_execPlanMulti[2];

         // no operand is modified on GPU so no need to copy back...
         else
            m_execPlan = &m_execPlanMulti[1];
      }

      /*! nothing is present on GPU... */
      else
         m_execPlan = &m_execPlanMulti[0];
   }
   if (m_execPlan == NULL || m_execPlan->calibrated == false)
      m_execPlan = &m_defPlan;
#endif
#endif
   assert(m_execPlan != NULL && m_execPlan->calibrated);

   size_t size = input1.total_nnz();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, input3, output, 0);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, input3, output, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, input3, output, 0);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, input3, output, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input1, input2, input3, output);
#endif
   case CPU_BACKEND:
      return CPU(input1, input2, input3, output);

   default:
      return CPU(input1, input2, input3, output);
   }
}





// ************************************************************************//
// ************************************************************************//
// -------------------------------    CPU   -------------------------------//
// ************************************************************************//
// ************************************************************************//

/*!
 *  Performs mapping on \em one sparse matrix on the \em CPU. Input is used as output.
 *
 *  \param input A sparse matrix which the mapping will be performed on. It will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(SparseMatrix<T>& input)
{
   DEBUG_TEXT_LEVEL1("MAP CPU\n")

   //Make sure we are properly synched with device data
   input.updateHostAndInvalidateDevice();

   T *values= input.get_values();
   size_t nnz= input.total_nnz();

   for(size_t i=0; i<nnz; i++)
   {
      values[i] = m_mapFunc->CPU(values[i]);
   }
}


/*!
 *  Performs mapping with \em one sparse matrix on the \em CPU. Seperate output is used.
 *
 *  \param input A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(SparseMatrix<T>& input, SparseMatrix<T>& output)
{
   DEBUG_TEXT_LEVEL1("MAP CPU\n")

   //Make sure we are properly synched with device data
   input.updateHost();
   output.invalidateDeviceData();

   T *input_values= input.get_values();
   T *output_values= output.get_values();

   size_t nnz= input.total_nnz();

   for(size_t i=0; i<nnz; i++)
   {
      output_values[i] = m_mapFunc->CPU(input_values[i]);
   }
}


/*!
 *  Performs mapping with \em two sparse matrices as input on the \em CPU. Seperate output is used.
 *
 *  \param input1 A sparse matrix which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& output)
{
   DEBUG_TEXT_LEVEL1("MAP CPU\n")

   //Make sure we are properly synched with device data
   input1.updateHost();
   input2.updateHost();
   output.invalidateDeviceData();

   T *input1_values= input1.get_values();
   T *input2_values= input2.get_values();
   T *output_values= output.get_values();

   size_t nnz= input1.total_nnz();

   for(size_t i=0; i<nnz; i++)
   {
      output_values[i] = m_mapFunc->CPU(input1_values[i],input2_values[i]);
   }
}




/*!
 *  Performs mapping with \em two sparse matrices as input on the \em CPU. Seperate output is used.
 *
 *  \param input1 A sparse matrix which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param input3 A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CPU(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& input3, SparseMatrix<T>& output)
{
   DEBUG_TEXT_LEVEL1("MAP CPU\n")

   //Make sure we are properly synched with device data
   input1.updateHost();
   input2.updateHost();
   input3.updateHost();
   output.invalidateDeviceData();

   T *input1_values= input1.get_values();
   T *input2_values= input2.get_values();
   T *input3_values= input3.get_values();
   T *output_values= output.get_values();

   size_t nnz= input1.total_nnz();

   for(size_t i=0; i<nnz; i++)
   {
      output_values[i] = m_mapFunc->CPU(input1_values[i],input2_values[i], input3_values[i]);
   }
}






// ************************************************************************//
// ************************************************************************//
// ************************************************************************//
// ************************************************************************//
// -------------------------------   OPENMP  ------------------------------//
// ************************************************************************//
// ************************************************************************//
// ************************************************************************//
// ************************************************************************//


#ifdef SKEPU_OPENMP

/*!
 *  Performs mapping on \em one sparse matrix on the \em OpenMP. Input is used as output.
 *
 *  \param input A sparse matrix which the mapping will be performed on. It will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(SparseMatrix<T>& input)
{
   DEBUG_TEXT_LEVEL1("MAP OpenMP\n")

   //Make sure we are properly synched with device data
   input.updateHostAndInvalidateDevice();

   T *values= input.get_values();
   size_t nnz= input.total_nnz();

   omp_set_num_threads(m_execPlan->numOmpThreads(nnz));

   #pragma omp parallel for
   for(size_t i=0; i<nnz; i++)
   {
      values[i] = m_mapFunc->CPU(values[i]);
   }
}


/*!
 *  Performs mapping with \em one sparse matrix on the \em OpenMP. Seperate output is used.
 *
 *  \param input A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(SparseMatrix<T>& input, SparseMatrix<T>& output)
{
   DEBUG_TEXT_LEVEL1("MAP OpenMP\n")

   //Make sure we are properly synched with device data
   input.updateHost();
   output.invalidateDeviceData();

   T *input_values= input.get_values();
   T *output_values= output.get_values();

   size_t nnz= input.total_nnz();

   omp_set_num_threads(m_execPlan->numOmpThreads(nnz));

   #pragma omp parallel for
   for(size_t i=0; i<nnz; i++)
   {
      output_values[i] = m_mapFunc->CPU(input_values[i]);
   }
}


/*!
 *  Performs mapping with \em two sparse matrices as input on the \em OpenMP. Seperate output is used.
 *
 *  \param input1 A sparse matrix which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& output)
{
   DEBUG_TEXT_LEVEL1("MAP OpenMP\n")

   //Make sure we are properly synched with device data
   input1.updateHost();
   input2.updateHost();
   output.invalidateDeviceData();

   T *input1_values= input1.get_values();
   T *input2_values= input2.get_values();
   T *output_values= output.get_values();

   size_t nnz= input1.total_nnz();

   omp_set_num_threads(m_execPlan->numOmpThreads(nnz));

   #pragma omp parallel for
   for(size_t i=0; i<nnz; i++)
   {
      output_values[i] = m_mapFunc->CPU(input1_values[i],input2_values[i]);
   }
}




/*!
 *  Performs mapping with \em two sparse matrices as input on the \em OpenMP. Seperate output is used.
 *
 *  \param input1 A sparse matrix which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param input3 A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::OMP(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& input3, SparseMatrix<T>& output)
{
   DEBUG_TEXT_LEVEL1("MAP OpenMP\n")

   //Make sure we are properly synched with device data
   input1.updateHost();
   input2.updateHost();
   input3.updateHost();
   output.invalidateDeviceData();

   T *input1_values= input1.get_values();
   T *input2_values= input2.get_values();
   T *input3_values= input3.get_values();
   T *output_values= output.get_values();

   size_t nnz= input1.total_nnz();

   omp_set_num_threads(m_execPlan->numOmpThreads(nnz));

   #pragma omp parallel for
   for(size_t i=0; i<nnz; i++)
   {
      output_values[i] = m_mapFunc->CPU(input1_values[i],input2_values[i], input3_values[i]);
   }
}

#endif



// ************************************************************************//
// ************************************************************************//
// ************************************************************************//
// ************************************************************************//
// -------------------------------    CUDA   ------------------------------//
// ************************************************************************//
// ************************************************************************//
// ************************************************************************//
// ************************************************************************//

#ifdef SKEPU_CUDA


/*!
 *  Applies the Map skeleton to \em one range of elements specified by iterators. Result is saved to a seperate output range.
 *  The calculations are performed by one host thread using \p one device with \em CUDA as backend.
 *
 *  The skeleton must have been created with a \em unary user function.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element in the range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param deviceID Integer specifying the which device to use.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::mapSingleThread_CU(SparseMatrix<T> &input, SparseMatrix<T> &output, unsigned int deviceID)
{
   cudaSetDevice(deviceID);

   // Setup parameters
   size_t n = input.total_nnz();
   BackEndParams bp=m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;

   numThreads = std::min(maxThreads, n);
   numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));

   // Copies the elements to the device
   typename SparseMatrix<T>::device_pointer_type_cu in_mem_p = input.updateDevice_CU( input.get_values(), n, deviceID, true);
   typename SparseMatrix<T>::device_pointer_type_cu out_mem_p;

   if(input.get_values()!=output.get_values()) // if not the same matrix
      out_mem_p = output.updateDevice_CU( output.get_values(), n, deviceID, false);
   else
      out_mem_p = in_mem_p;

   // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
   MapKernelUnary_CU<<<numBlocks,numThreads,0, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#else
   MapKernelUnary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#endif

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}


/*!
 *  Performs mapping on \em one sparse matrix on the \em CUDA. Input is used as output.
 *
 *  \param input A sparse matrix which the mapping will be performed on. It will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(SparseMatrix<T>& input, int useNumGPU)
{
   CU(input, input, useNumGPU);
}


/*!
 *  Performs mapping with \em one sparse matrix on the \em CUDA. Seperate output is used.
 *
 *  \param input A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(SparseMatrix<T>& input, SparseMatrix<T>& output, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapSingleThread_CU(input, output, 0);
   }
   else
   {
      size_t n = input.total_nnz();
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename SparseMatrix<T>::device_pointer_type_cu in_mem_p[MAX_GPU_DEVICES];
      typename SparseMatrix<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      T *inputValues = input.get_values();
      T *outputValues = output.get_values();

      bool sameOperands = (inputValues==outputValues);

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in_mem_p[i] = input.updateDevice_CU((inputValues+i*numElemPerSlice), numElem,  i, false);

         if(!sameOperands) // if not the same matrix
            out_mem_p[i] = output.updateDevice_CU((outputValues+i*numElemPerSlice), numElem,  i, false);
         else
            out_mem_p[i] = in_mem_p[i];
      }

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         //Copy input now
         in_mem_p[i] = input.updateDevice_CU((inputValues+i*numElemPerSlice), numElem,  i, true);

         // Setup parameters
         BackEndParams bp=m_execPlan->find_(numElem);
         size_t maxBlocks = bp.maxBlocks;
         size_t maxThreads = bp.maxThreads;
         size_t numBlocks;
         size_t numThreads;

         numThreads = std::min(maxThreads, numElem);
         numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

#ifdef USE_PINNED_MEMORY
         MapKernelUnary_CU<<<numBlocks,numThreads,0, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#else
         MapKernelUnary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#endif

         // Copy back result (synchronizes)
         out_mem_p[i]->changeDeviceData();
      }

      cudaSetDevice(m_environment->bestCUDADevID);
   }
}

/*!
 *  Applies the Map skeleton to \em one range of elements specified by iterators. Result is saved to a seperate output range.
 *  The calculations are performed by one host thread using \p one device with \em CUDA as backend.
 *
 *  The skeleton must have been created with a \em unary user function.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element in the range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param deviceID Integer specifying the which device to use.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::mapSingleThread_CU(SparseMatrix<T> &input1, SparseMatrix<T> &input2, SparseMatrix<T> &output, unsigned int deviceID)
{
   cudaSetDevice(deviceID);

   // Setup parameters
   size_t n = input1.total_nnz();
   BackEndParams bp=m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;

   numThreads = std::min(maxThreads, n);
   numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));


   // Copies the elements to the device
   typename SparseMatrix<T>::device_pointer_type_cu in1_mem_p = input1.updateDevice_CU( input1.get_values(), n, deviceID, true);
   typename SparseMatrix<T>::device_pointer_type_cu in2_mem_p = input2.updateDevice_CU( input2.get_values(), n, deviceID, true);
   typename SparseMatrix<T>::device_pointer_type_cu out_mem_p;


   out_mem_p = output.updateDevice_CU( output.get_values(), n, deviceID, false);

   // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
   MapKernelBinary_CU<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#else
   MapKernelBinary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#endif

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}


/*!
 *  Performs mapping with \em two sparse matrices as input on the \em CUDA. Seperate output is used.
 *
 *  \param input1 A sparse matrix which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& output, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapSingleThread_CU(input1, input2, output, 0);
   }
   else
   {
      size_t n = input1.total_nnz();
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename SparseMatrix<T>::device_pointer_type_cu in1_mem_p[MAX_GPU_DEVICES];
      typename SparseMatrix<T>::device_pointer_type_cu in2_mem_p[MAX_GPU_DEVICES];
      typename SparseMatrix<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      T *input1Values = input1.get_values();
      T *input2Values = input2.get_values();
      T *outputValues = output.get_values();

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in1_mem_p[i] = input1.updateDevice_CU((input1Values+i*numElemPerSlice), numElem,  i, false);
         in2_mem_p[i] = input2.updateDevice_CU((input2Values+i*numElemPerSlice), numElem,  i, false);

         out_mem_p[i] = output.updateDevice_CU((outputValues+i*numElemPerSlice), numElem,  i, false);
      }

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         //Copy input now
         in1_mem_p[i] = input1.updateDevice_CU((input1Values+i*numElemPerSlice), numElem,  i, true);
         in2_mem_p[i] = input2.updateDevice_CU((input2Values+i*numElemPerSlice), numElem,  i, true);

         // Setup parameters
         BackEndParams bp=m_execPlan->find_(numElem);
         size_t maxBlocks = bp.maxBlocks;
         size_t maxThreads = bp.maxThreads;
         size_t numBlocks;
         size_t numThreads;

         numThreads = std::min(maxThreads, numElem);
         numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

#ifdef USE_PINNED_MEMORY
         MapKernelBinary_CU<<<numBlocks,numThreads,0, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#else
         MapKernelBinary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#endif

         // Copy back result (synchronizes)
         out_mem_p[i]->changeDeviceData();
      }

      cudaSetDevice(m_environment->bestCUDADevID);
   }

}



/*!
 *  Applies the Map skeleton to \em one range of elements specified by iterators. Result is saved to a seperate output range.
 *  The calculations are performed by one host thread using \p one device with \em CUDA as backend.
 *
 *  The skeleton must have been created with a \em unary user function.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element in the range.
 *  \param outputBegin An iterator to the first element of the output range.
 *  \param deviceID Integer specifying the which device to use.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::mapSingleThread_CU(SparseMatrix<T> &input1, SparseMatrix<T> &input2, SparseMatrix<T> &input3, SparseMatrix<T> &output, unsigned int deviceID)
{
   cudaSetDevice(deviceID);

   // Setup parameters
   size_t n = input1.total_nnz();
   BackEndParams bp=m_execPlan->find_(n);
   size_t maxBlocks = bp.maxBlocks;
   size_t maxThreads = bp.maxThreads;
   size_t numBlocks;
   size_t numThreads;

   numThreads = std::min(maxThreads, n);
   numBlocks = std::max((size_t)1, std::min( (n/numThreads + (n%numThreads == 0 ? 0:1)), maxBlocks));



   // Copies the elements to the device
   typename SparseMatrix<T>::device_pointer_type_cu in1_mem_p = input1.updateDevice_CU( input1.get_values(), n, deviceID, true);
   typename SparseMatrix<T>::device_pointer_type_cu in2_mem_p = input2.updateDevice_CU( input2.get_values(), n, deviceID, true);
   typename SparseMatrix<T>::device_pointer_type_cu in3_mem_p = input3.updateDevice_CU( input3.get_values(), n, deviceID, true);
   typename SparseMatrix<T>::device_pointer_type_cu out_mem_p;


   out_mem_p = output.updateDevice_CU( output.get_values(), n, deviceID, false);

   // Launches the kernel (asynchronous)
#ifdef USE_PINNED_MEMORY
   MapKernelTrinary_CU<<<numBlocks,numThreads, 0, (m_environment->m_devices_CU.at(deviceID)->m_streams[0])>>>(*m_mapFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), in3_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#else
   MapKernelTrinary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in1_mem_p->getDeviceDataPointer(), in2_mem_p->getDeviceDataPointer(), in3_mem_p->getDeviceDataPointer(), out_mem_p->getDeviceDataPointer(), n);
#endif

   // Make sure the data is marked as changed by the device
   out_mem_p->changeDeviceData();
}




/*!
 *  Performs mapping with \em two sparse matrices as input on the \em CUDA. Seperate output is used.
 *
 *  \param input1 A sparse matrix which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param input3 A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CU(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& input3, SparseMatrix<T>& output, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP CUDA\n")

   size_t numDevices = m_environment->m_devices_CU.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }
   if(numDevices < 2)
   {
      mapSingleThread_CU(input1, input2, input3, output, 0);
   }
   else
   {
      size_t n = input1.total_nnz();
      size_t numElemPerSlice = n / numDevices;
      size_t rest = n % numDevices;

      typename SparseMatrix<T>::device_pointer_type_cu in1_mem_p[MAX_GPU_DEVICES];
      typename SparseMatrix<T>::device_pointer_type_cu in2_mem_p[MAX_GPU_DEVICES];
      typename SparseMatrix<T>::device_pointer_type_cu in3_mem_p[MAX_GPU_DEVICES];
      typename SparseMatrix<T>::device_pointer_type_cu out_mem_p[MAX_GPU_DEVICES];

      T *input1Values = input1.get_values();
      T *input2Values = input2.get_values();
      T *input3Values = input3.get_values();
      T *outputValues = output.get_values();

      // First create CUDA memory if not created already.
      for(size_t i = 0; i < numDevices; ++i)
      {
         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         in1_mem_p[i] = input1.updateDevice_CU((input1Values+i*numElemPerSlice), numElem,  i, false);
         in2_mem_p[i] = input2.updateDevice_CU((input2Values+i*numElemPerSlice), numElem,  i, false);
         in3_mem_p[i] = input3.updateDevice_CU((input3Values+i*numElemPerSlice), numElem,  i, false);

         out_mem_p[i] = output.updateDevice_CU((outputValues+i*numElemPerSlice), numElem,  i, false);
      }

      // Fill out argument struct with right information and start threads.
      for(size_t i = 0; i < numDevices; ++i)
      {
         cudaSetDevice(i);

         size_t numElem;
         if(i == numDevices-1)
            numElem = numElemPerSlice+rest;
         else
            numElem = numElemPerSlice;

         //Copy input now
         in1_mem_p[i] = input1.updateDevice_CU((input1Values+i*numElemPerSlice), numElem,  i, true);
         in2_mem_p[i] = input2.updateDevice_CU((input2Values+i*numElemPerSlice), numElem,  i, true);
         in3_mem_p[i] = input3.updateDevice_CU((input3Values+i*numElemPerSlice), numElem,  i, true);

         // Setup parameters
         BackEndParams bp=m_execPlan->find_(numElem);
         size_t maxBlocks = bp.maxBlocks;
         size_t maxThreads = bp.maxThreads;
         size_t numBlocks;
         size_t numThreads;

         numThreads = std::min(maxThreads, numElem);
         numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

#ifdef USE_PINNED_MEMORY
         MapKernelTrinary_CU<<<numBlocks,numThreads,0, (m_environment->m_devices_CU.at(i)->m_streams[0])>>>(*m_mapFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), in3_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#else
         MapKernelTrinary_CU<<<numBlocks,numThreads>>>(*m_mapFunc, in1_mem_p[i]->getDeviceDataPointer(), in2_mem_p[i]->getDeviceDataPointer(), in3_mem_p[i]->getDeviceDataPointer(), out_mem_p[i]->getDeviceDataPointer(), numElem);
#endif

         // Copy back result (synchronizes)
         out_mem_p[i]->changeDeviceData();
      }

      cudaSetDevice(m_environment->bestCUDADevID);
   }
}

#endif












// ************************************************************************//
// ************************************************************************//
// ************************************************************************//
// ************************************************************************//
// -------------------------------    OpenCL   ------------------------------//
// ************************************************************************//
// ************************************************************************//
// ************************************************************************//
// ************************************************************************//

#ifdef SKEPU_OPENCL




/*!
 *  Performs mapping on \em one sparse matrix on the \em OpenCL. Input is used as output.
 *
 *  \param input A sparse matrix which the mapping will be performed on. It will be overwritten with the result.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(SparseMatrix<T>& input, int useNumGPU)
{
   CL(input, input, useNumGPU);
}


/*!
 *  Performs mapping with \em one sparse matrix on the \em OpenCL. Seperate output is used.
 *
 *  \param input A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(SparseMatrix<T>& input, SparseMatrix<T>& output, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP OpenCL\n")

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }

   cl_int err;

   // Divide the elements amongst the devices
   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   size_t n = input.total_nnz();
   size_t numElemPerSlice = n / numDevices;
   size_t rest = n % numDevices;

   typename SparseMatrix<T>::device_pointer_type_cl in_mem_p[MAX_GPU_DEVICES];
   typename SparseMatrix<T>::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   T *inputValues = input.get_values();
   T *outputValues = output.get_values();

   bool sameOperands = (inputValues==outputValues);

   // First create CUDA memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      in_mem_p[i] = input.updateDevice_CL((inputValues+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, false);

      if(!sameOperands) // if not the same matrix
         out_mem_p[i] = output.updateDevice_CL((outputValues+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, false);
      else
         out_mem_p[i] = in_mem_p[i];
   }

   // Fill out argument struct with right information and start threads.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      //Copy input now
      in_mem_p[i] = input.updateDevice_CL((inputValues+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, true);

      // Setup parameters
      BackEndParams bp;
      bp=m_execPlan->find_(numElem);
      size_t maxBlocks = bp.maxBlocks;
      size_t maxThreads = bp.maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      numThreads = std::min(maxThreads, (size_t)numElem);
      numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

      cl_mem in_p = in_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 2, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 3, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}



/*!
 *  Performs mapping with \em two sparse matrices as input on the \em OpenCL. Seperate output is used.
 *
 *  \param input1 A sparse matrix which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& output, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP OpenCL\n")

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }

   cl_int err;

   // Divide the elements amongst the devices
   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   size_t n = input1.total_nnz();
   size_t numElemPerSlice = n / numDevices;
   size_t rest = n % numDevices;

   typename SparseMatrix<T>::device_pointer_type_cl in1_mem_p[MAX_GPU_DEVICES];
   typename SparseMatrix<T>::device_pointer_type_cl in2_mem_p[MAX_GPU_DEVICES];
   typename SparseMatrix<T>::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   T *input1Values = input1.get_values();
   T *input2Values = input2.get_values();
   T *outputValues = output.get_values();

   // First create CUDA memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      in1_mem_p[i] = input1.updateDevice_CL((input1Values+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, false);
      in2_mem_p[i] = input2.updateDevice_CL((input2Values+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, false);

      out_mem_p[i] = output.updateDevice_CL((outputValues+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, false);
   }

   // Fill out argument struct with right information and start threads.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      //Copy input now
      in1_mem_p[i] = input1.updateDevice_CL((input1Values+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, true);
      in2_mem_p[i] = input2.updateDevice_CL((input2Values+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, true);

      // Setup parameters
      BackEndParams bp;
      bp=m_execPlan->find_(numElem);
      size_t maxBlocks = bp.maxBlocks;
      size_t maxThreads = bp.maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      numThreads = std::min(maxThreads, (size_t)numElem);
      numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

      cl_mem in1_p = in1_mem_p[i]->getDeviceDataPointer();
      cl_mem in2_p = in2_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 3, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 4, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}






/*!
 *  Performs mapping with \em two sparse matrices as input on the \em OpenCL. Seperate output is used.
 *
 *  \param input1 A sparse matrix which the mapping will be performed on.
 *  \param input2 A sparse matrix which the mapping will be performed on.
 *  \param input3 A sparse matrix which the mapping will be performed on.
 *  \param output The result sparse matrix, will be overwritten with the result and resized if needed.
 *  \param useNumGPU Integer specifying how many devices to use. 0 = implementation decides.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::CL(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& input3, SparseMatrix<T>& output, int useNumGPU)
{
   DEBUG_TEXT_LEVEL1("MAP OpenCL\n")

   size_t numDevices = m_kernels_CL.size();

   if(useNumGPU > 0) // if user has specified the desired number of GPUs to be used
   {
      numDevices = (useNumGPU<numDevices)? useNumGPU : numDevices;
   }

   cl_int err;

   // Divide the elements amongst the devices
   typename MapFunc::CONST_TYPE const1 = m_mapFunc->getConstant();

   size_t n = input1.total_nnz();
   size_t numElemPerSlice = n / numDevices;
   size_t rest = n % numDevices;

   typename SparseMatrix<T>::device_pointer_type_cl in1_mem_p[MAX_GPU_DEVICES];
   typename SparseMatrix<T>::device_pointer_type_cl in2_mem_p[MAX_GPU_DEVICES];
   typename SparseMatrix<T>::device_pointer_type_cl in3_mem_p[MAX_GPU_DEVICES];
   typename SparseMatrix<T>::device_pointer_type_cl out_mem_p[MAX_GPU_DEVICES];

   T *input1Values = input1.get_values();
   T *input2Values = input2.get_values();
   T *input3Values = input3.get_values();
   T *outputValues = output.get_values();

   // First create CUDA memory if not created already.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      in1_mem_p[i] = input1.updateDevice_CL((input1Values+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, false);
      in2_mem_p[i] = input2.updateDevice_CL((input2Values+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, false);
      in3_mem_p[i] = input3.updateDevice_CL((input3Values+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, false);

      out_mem_p[i] = output.updateDevice_CL((outputValues+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, false);
   }

   // Fill out argument struct with right information and start threads.
   for(size_t i = 0; i < numDevices; ++i)
   {
      size_t numElem;
      if(i == numDevices-1)
         numElem = numElemPerSlice+rest;
      else
         numElem = numElemPerSlice;

      //Copy input now
      in1_mem_p[i] = input1.updateDevice_CL((input1Values+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, true);
      in2_mem_p[i] = input2.updateDevice_CL((input2Values+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, true);
      in3_mem_p[i] = input3.updateDevice_CL((input3Values+i*numElemPerSlice), numElem,  m_kernels_CL.at(i).second, true);

      // Setup parameters
      BackEndParams bp;
      bp=m_execPlan->find_(numElem);
      size_t maxBlocks = bp.maxBlocks;
      size_t maxThreads = bp.maxThreads;
      size_t numBlocks;
      size_t numThreads;
      size_t globalWorkSize[1];
      size_t localWorkSize[1];

      numThreads = std::min(maxThreads, (size_t)numElem);
      numBlocks = std::max((size_t)1, std::min( (numElem/numThreads + (numElem%numThreads == 0 ? 0:1)), maxBlocks));

      cl_mem in1_p = in1_mem_p[i]->getDeviceDataPointer();
      cl_mem in2_p = in2_mem_p[i]->getDeviceDataPointer();
      cl_mem in3_p = in3_mem_p[i]->getDeviceDataPointer();
      cl_mem out_p = out_mem_p[i]->getDeviceDataPointer();

      cl_kernel kernel = m_kernels_CL.at(i).first;

      // Sets the kernel arguments
      clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in1_p);
      clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in2_p);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&in3_p);
      clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&out_p);
      clSetKernelArg(kernel, 4, sizeof(size_t), (void*)&numElem);
      clSetKernelArg(kernel, 5, sizeof(typename MapFunc::CONST_TYPE), (void*)&const1);

      globalWorkSize[0] = numBlocks * numThreads;
      localWorkSize[0] = numThreads;

      // Launches the kernel (asynchronous)
      err = clEnqueueNDRangeKernel(m_kernels_CL.at(i).second->getQueue(), kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error launching kernel!!\n");
      }

      // Make sure the data is marked as changed by the device
      out_mem_p[i]->changeDeviceData();
   }
}

#endif

}
