/*! \file reduce.inl
 *  \brief Contains the definitions of non-backend specific member functions for the Reduce skeleton.
 */

#ifdef SKEPU_OPENMP
#include <omp.h>
#endif

namespace skepu
{

/*!
 *  When creating an instance of the Reduce skeleton, a pointer to a binary user function must be provided.
 *  Also the Environment is set and if \p SKEPU_OPENCL is defined, the appropriate OpenCL program
 *  and kernel are created. Also creates a default execution plan which the skeleton will use if no other is
 *  specified.
 *
 *  \param reduceFunc A pointer to a valid binary user function. Will be deleted in the destructor.
 */
template <typename ReduceFunc>
Reduce<ReduceFunc, ReduceFunc>::Reduce(ReduceFunc* reduceFunc)
{
   if(reduceFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type! Reduce operation require binary user function.\n");
   }

   BackEndParams bp;

   m_reduceFunc = reduceFunc;
   m_environment = Environment<int>::getInstance();

#if defined(SKEPU_OPENCL) && !defined(SKEPU_CUDA) && SKEPU_NUMGPU == 1
   bp.backend=CL_BACKEND;
#elif defined(SKEPU_OPENCL) && !defined(SKEPU_CUDA) && SKEPU_NUMGPU != 1
   bp.backend=CLM_BACKEND;
#elif !defined(SKEPU_OPENCL) && defined(SKEPU_CUDA) && SKEPU_NUMGPU == 1
   bp.backend=CU_BACKEND;
#elif !defined(SKEPU_OPENCL) && defined(SKEPU_CUDA) && SKEPU_NUMGPU != 1
   bp.backend=CUM_BACKEND;
#elif defined(SKEPU_OPENCL) && defined(SKEPU_CUDA) && SKEPU_NUMGPU == 1
   bp.backend=CL_BACKEND;
#elif defined(SKEPU_OPENCL) && defined(SKEPU_CUDA) && SKEPU_NUMGPU != 1
   bp.backend=CLM_BACKEND;
#elif !defined(SKEPU_OPENCL) && !defined(SKEPU_CUDA)

#if defined(SKEPU_OPENMP)
   bp.backend=OMP_BACKEND;
#else
   bp.backend=CPU_BACKEND;
#endif

#endif

#ifdef SKEPU_OPENCL
   bp.maxThreads = m_environment->m_devices_CL.at(0)->getMaxThreads();
   bp.maxBlocks = m_environment->m_devices_CL.at(0)->getMaxBlocks();
#endif

#ifdef SKEPU_CUDA
   cudaDeviceID = Environment<int>::getInstance()->bestCUDADevID;

   bp.maxThreads = m_environment->m_devices_CU.at(0)->getMaxThreads();
   bp.maxBlocks = m_environment->m_devices_CU.at(0)->getMaxBlocks();
#endif

#ifdef SKEPU_OPENMP
#ifdef SKEPU_OPENMP_THREADS
   bp.numOmpThreads = SKEPU_OPENMP_THREADS;
#else
   bp.numOmpThreads = omp_get_max_threads();
#endif
#endif

   m_defPlan.calibrated = true;
   m_defPlan.add(1, 100000000, bp);
   m_execPlan = &m_defPlan;
   m_execPlanMulti = NULL;
   
#ifdef SKEPU_OPENCL
   createOpenCLProgram();
#endif
}

/*!
 *  When the Reduce skeleton is destroyed, it deletes the user function it was created with.
 */
template <typename ReduceFunc>
Reduce<ReduceFunc, ReduceFunc>::~Reduce()
{
   delete  m_reduceFunc;
}

/*!
 *  Performs the Reduction on a Vector. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A vector which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::operator()(Vector<T>& input)
{
// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      if(input.isVectorOnDevice_CU(cudaDeviceID))
      {
         if(input.isModified_CU(cudaDeviceID))
            m_execPlan = &(m_execPlanMulti[2]);
         else
            m_execPlan = &(m_execPlanMulti[1]);
      }
      else
         m_execPlan = &(m_execPlanMulti[0]);
   }
   if (m_execPlan == NULL)
      m_execPlan = &m_defPlan;
   if (m_execPlan->calibrated == false)
      m_execPlan = &m_defPlan;
#endif
   assert(m_execPlan != NULL && m_execPlan->calibrated);

   int size = input.size();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, SKEPU_NUMGPU);
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
 *  Performs the Reduction on a Matrix. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::operator()(Matrix<T>& input)
{
// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      if(input.isMatrixOnDevice_CU(cudaDeviceID))
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
   assert(m_execPlan != NULL && m_execPlan->calibrated);

   int size = input.size();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, SKEPU_NUMGPU);
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
 *  Performs the Reduction on a Matrix either row or column-wise. Returns a \em skepu Vector containing reduction
 *  results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the reduction will be performed on.
 *  \param reducePolicy The policy specifying how reduction will be performed, can be either REDUCE_ROW_WISE_ONLY of REDUCE_COL_WISE_ONLY
 *  \return A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
Vector<T> Reduce<ReduceFunc, ReduceFunc>::operator()(Matrix<T>& input, ReducePolicy reducePolicy)
{
// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      if(input.isMatrixOnDevice_CU(cudaDeviceID))
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
   assert(m_execPlan != NULL && m_execPlan->calibrated);

   int size = input.size();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, reducePolicy, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, reducePolicy, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, reducePolicy, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, reducePolicy, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, reducePolicy);
#endif
   case CPU_BACKEND:
      return CPU(input, reducePolicy);

   default:
      return CPU(input, reducePolicy);
   }
}




/*!
 *  Performs the Reduction on non-zero elements of a SparseMatrix. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A sparse matrix which the reduction will be performed on.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename T>
T Reduce<ReduceFunc, ReduceFunc>::operator()(SparseMatrix<T>& input)
{
// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      if(input.isSparseMatrixOnDevice_CU(cudaDeviceID))
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
   assert(m_execPlan != NULL && m_execPlan->calibrated);

   int size = input.total_nnz();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, SKEPU_NUMGPU);
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
 *  Performs the Reduction on non-zero elements of a sparse Matrix either row or column-wise. Returns a \em skepu Vector containing reduction
 *  results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A sparse matrix which the reduction will be performed on.
 *  \param reducePolicy The policy specifying how reduction will be performed, can be either REDUCE_ROW_WISE_ONLY of REDUCE_COL_WISE_ONLY
 *  \return A \em skepu Vector containing reduction results either row-wise or column-wise depending upon supplied \em ReducePolicy.
 */
template <typename ReduceFunc>
template <typename T>
Vector<T> Reduce<ReduceFunc, ReduceFunc>::operator()(SparseMatrix<T>& input, ReducePolicy reducePolicy)
{
// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      if(input.isSparseMatrixOnDevice_CU(cudaDeviceID))
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
   assert(m_execPlan != NULL && m_execPlan->calibrated);

   int size = input.total_nnz();
   BackEnd backEnd = m_execPlan->find(size);

   if(input.m_transMatrix && reducePolicy==REDUCE_COL_WISE_ONLY)
   {
      SKEPU_ERROR("Cannot apply column-wise reduction on a matrix which is already in CSC format.\n");
   }

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, reducePolicy, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, reducePolicy, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, reducePolicy, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, reducePolicy, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, reducePolicy);
#endif
   case CPU_BACKEND:
      return CPU(input, reducePolicy);

   default:
      return CPU(input, reducePolicy);
   }
}






/*!
 *  Performs the Reduction on a range of elements. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \return The scalar result of the reduction performed.
 */
template <typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type Reduce<ReduceFunc, ReduceFunc>::operator()(InputIterator inputBegin, InputIterator inputEnd)
{
   int size = inputEnd - inputBegin;
   
   m_execPlan = &m_defPlan;
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(inputBegin, inputEnd);
#endif
   case CPU_BACKEND:
      return CPU(inputBegin, inputEnd);

   default:
      return CPU(inputBegin, inputEnd);
   }
}

}

