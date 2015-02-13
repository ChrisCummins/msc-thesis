/*! \file reduce_2d.inl
 *  \brief Contains the definitions of non-backend specific member functions for the 2DReduce skeleton.
 */

#ifdef SKEPU_OPENMP
#include <omp.h>
#endif

namespace skepu
{


/*!
 *  When creating an instance of the 2D Reduce (First row-wise then column-wise) skeleton,
 *  a pointer to a binary user function must be provided.
 *  Also the Environment is set and if \p SKEPU_OPENCL is defined, the appropriate OpenCL program
 *  and kernel are created. Also creates a default execution plan which the skeleton will use if no other is
 *  specified.
 *
 *  \param reduceFuncRowWise A pointer to a valid binary user function used for row-wise reduction. Will be deleted in the destructor.
 *  \param reduceFuncColWise A pointer to a valid binary user function used for columns-wise reduction. Will be deleted in the destructor.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
Reduce<ReduceFuncRowWise, ReduceFuncColWise>::Reduce(ReduceFuncRowWise* reduceFuncRowWise, ReduceFuncColWise* reduceFuncColWise)
{
   if( (reduceFuncRowWise->funcType != BINARY) || (reduceFuncColWise->funcType != BINARY) )
   {
      SKEPU_ERROR("Wrong operator type! 2D Reduce operation require binary user functions.\n");
   }

   BackEndParams bp;

   m_reduceFuncRowWise = reduceFuncRowWise;
   m_reduceFuncColWise = reduceFuncColWise;

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
 *  When the 2D Reduce skeleton is destroyed, it deletes the user functions it was created with.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
Reduce<ReduceFuncRowWise, ReduceFuncColWise>::~Reduce()
{
   delete  m_reduceFuncRowWise;
   delete  m_reduceFuncColWise;
}



/*!
 *  Performs the 2D Reduction (First row-wise then column-wise) on a whole Matrix. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix on which the 2D reduction will be performed on.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::operator()(Matrix<T>& input)
{
   if( (input.total_rows()<2) || (input.total_cols()<2))
   {
      SKEPU_ERROR("Error: The 2D reduction operator can be applied on a matrix with atleast 2 rows and 2 columns.\n");
   }

//      assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   /*! It might be the case that in a complex program suchg as SPH, some skeleton obj use default exec plan as are not perf relevant... */
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
 *  Performs the 2D Reduction (First row-wise then column-wise) on a whole Matrix. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix on which the 2D reduction will be performed on.
 *  \return The scalar result of the 2D reduction performed.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise>
template <typename T>
T Reduce<ReduceFuncRowWise, ReduceFuncColWise>::operator()(SparseMatrix<T>& input)
{
   if(input.total_nnz()==0) // if no non-zero element, return T()
      return T();

   if( (input.total_rows()<2) || (input.total_cols()<2) || (input.total_nnz()<2))
   {
      SKEPU_ERROR("Error: The 2D reduction operator can be applied on a sparse matrix with atleast 2 rows, 2 columns and 2 non-zero elements.\n");
   }
   
//      assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   /*! It might be the case that in a complex program suchg as SPH, some skeleton obj use default exec plan as are not perf relevant... */
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



}

