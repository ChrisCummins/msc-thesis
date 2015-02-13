/*! \file 2dmapoverlap.inl
 *  \brief Contains the definitions of non-backend specific member functions for the MapOverlap2D skeleton.
 */

#ifdef SKEPU_OPENMP
#include <omp.h>
#endif


#define BLOCK_DIM 16


namespace skepu
{

/*!
 *  When creating an instance of the MapOverlap2D skeleton, a pointer to an overlap user function must be provided.
 *  Also the environment is set and if \p SKEPU_OPENCL is defined, the appropriate OpenCL program
 *  and kernel are created. Also creates a default execution plan which the skeleton will use if no other is
 *  specified.
 *
 *  \param mapOverlapFunc A pointer to a valid overlap user function. Will be deleted in the destructor.
 */
template <typename MapOverlap2DFunc>
MapOverlap2D<MapOverlap2DFunc>::MapOverlap2D(MapOverlap2DFunc* mapOverlapFunc)
{
   if(mapOverlapFunc->funcType != OVERLAP_2D)
   {
      SKEPU_ERROR("Wrong operator type! Should use OVERLAP_FUNC_2D_STR macro\n");
   }
   
   BackEndParams bp;
   m_mapOverlapFunc = mapOverlapFunc;
   m_environment = Environment<int>::getInstance();

#if defined(SKEPU_OPENCL) && !defined(SKEPU_CUDA) && SKEPU_NUMGPU == 1
   bp.backend = CL_BACKEND;
#elif defined(SKEPU_OPENCL) && !defined(SKEPU_CUDA) && SKEPU_NUMGPU != 1
   bp.backend = CLM_BACKEND;
#elif !defined(SKEPU_OPENCL) && defined(SKEPU_CUDA) && SKEPU_NUMGPU == 1
   bp.backend = CU_BACKEND;
#elif !defined(SKEPU_OPENCL) && defined(SKEPU_CUDA) && SKEPU_NUMGPU != 1
   bp.backend = CUM_BACKEND;
#elif defined(SKEPU_OPENCL) && defined(SKEPU_CUDA) && SKEPU_NUMGPU == 1
   bp.backend = CL_BACKEND;
#elif defined(SKEPU_OPENCL) && defined(SKEPU_CUDA) && SKEPU_NUMGPU != 1
   bp.backend = CLM_BACKEND;
#elif !defined(SKEPU_OPENCL) && !defined(SKEPU_CUDA)

#if defined(SKEPU_OPENMP)
   bp.backend = OMP_BACKEND;
#else
   bp.backend = CPU_BACKEND;
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

#ifdef SKEPU_OPENCL
   createOpenCLProgram();
#endif
}

/*!
 *  When the MapOverlap2D skeleton is destroyed, it deletes the user function it was created with.
 */
template <typename MapOverlap2DFunc>
MapOverlap2D<MapOverlap2DFunc>::~MapOverlap2D()
{
   delete  m_mapOverlapFunc;
}

/*!
 *  Performs the MapOverlap2D on a whole Matrix where actual filter is specified in a user-function. With itself as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the mapping will be performed on. It will be overwritten with the result.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::operator()(Matrix<T>& input)
{
#ifdef USE_MULTI_EXEC_PLAN
//      assert(m_execPlanMulti != NULL);
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
#endif
   assert(m_execPlan != NULL && m_execPlan->calibrated);
   
   size_t size = input.size();
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
 *  Performs the MapOverlap2D on a whole Matrix where actual filter is specified in a user-function. With a seperate Matrix as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::operator()(Matrix<T>& input, Matrix<T>& output)
{
#ifdef USE_MULTI_EXEC_PLAN
//      assert(m_execPlanMulti != NULL);
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
   else if (m_execPlan == NULL)
      m_execPlan = &m_defPlan;
#endif
#endif
   assert(m_execPlan != NULL && m_execPlan->calibrated);
   
   size_t size = input.size();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, SKEPU_NUMGPU);
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
 *  Performs the MapOverlap2D based on provided filter and input neighbouring elements on a whole Matrix.
 *  With a seperate Matrix as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the mapping will be performed on. It should include padded data as well considering the filter size
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param filter The filter matrix which will be applied for each element in the output.
 *  \param useTiling The boolean flag that specify whether to use tiling optimizations especially when using the CUDA implementation.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::operator()(Matrix<T>& input, Matrix<T>& output, Matrix<T>& filter, bool useTiling)
{
#ifdef USE_MULTI_EXEC_PLAN
//      assert(m_execPlanMulti != NULL);
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
   else if (m_execPlan == NULL)
      m_execPlan = &m_defPlan;
#endif
#endif
   assert(m_execPlan != NULL && m_execPlan->calibrated);
   
   size_t size = input.size();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output,filter,useTiling, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output, filter,useTiling, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, filter, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, filter, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, output, filter);
#endif
   case CPU_BACKEND:
      return CPU(input, output, filter);

   default:
      return CPU(input, output, filter);
   }
}




/*!
 *  Performs the MapOverlap2D on a whole Matrix based on neighbouring elements by taking average of all neighbours. With a seperate Matrix as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the mapping will be performed on. It should include padded data as well considering the filter size
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param filter_rows The number of rows used as neighbouring elements to calculate new value for each output element.
 *  \param filter_cols The number of columns used as neighbouring elements to calculate new value for each output element.
 *  \param useTiling The boolean flag that specify whether to use tiling optimizations especially when using the CUDA implementation.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::operator()(Matrix<T>& input, Matrix<T>& output, size_t filter_rows, size_t filter_cols, bool useTiling)
{
#ifdef USE_MULTI_EXEC_PLAN
//      assert(m_execPlanMulti != NULL);
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
   else if (m_execPlan == NULL)
      m_execPlan = &m_defPlan;
#endif
#endif
   assert(m_execPlan != NULL && m_execPlan->calibrated);
   
   size_t size = input.size();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output,filter_rows, filter_cols,useTiling, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output, filter_rows, filter_cols,useTiling, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, filter_rows, filter_cols, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, filter_rows, filter_cols, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, output, filter_rows, filter_cols);
#endif
   case CPU_BACKEND:
      return CPU(input, output, filter_rows, filter_cols);

   default:
      return CPU(input, output, filter_rows, filter_cols);
   }
}

}

