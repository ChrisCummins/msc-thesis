/*! \file generate.inl
 *  \brief Contains the definitions of non-backend specific member functions for the Generate skeleton.
 */

#ifdef SKEPU_OPENMP
#include <omp.h>
#endif

namespace skepu
{

/*!
 *  When creating an instance of the Generate skeleton, a pointer to a generate user function must be provided.
 *  Also the environment is set, and if \p SKEPU_OPENCL is defined, the appropriate OpenCL program
 *  and kernel are created. Also creates a default execution plan which the skeleton will use if no other is
 *  specified.
 *
 *  \param generateFunc A pointer to a valid user function. Will be deleted in the destructor.
 */
template <typename GenerateFunc>
Generate<GenerateFunc>::Generate(GenerateFunc* generateFunc)
{
   BackEndParams bp;
   m_generateFunc = generateFunc;
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
   m_execPlanMulti = NULL;

#ifdef SKEPU_OPENCL
   createOpenCLProgram();
#endif
}

/*!
 *  When the Generate skeleton is destroyed, it deletes the user function it was created with.
 */
template <typename GenerateFunc>
Generate<GenerateFunc>::~Generate()
{
   delete m_generateFunc;
}

/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output vector.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param numElements The number of elements to be generated.
 *  \param output The output vector which will be overwritten with the generated values.
 */
template <typename GenerateFunc>
template <typename T>
void Generate<GenerateFunc>::operator()(size_t numElements, Vector<T>& output)
{
   if(m_generateFunc->funcType != GENERATE)
   {
      SKEPU_ERROR("Wrong operator type for Generate! Should use GENERATE_FUNC macro\n");
      return;
   }
   
   size_t size = numElements;
   
//      assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      m_execPlan = &m_execPlanMulti[0];
   }
   if (m_execPlan == NULL || m_execPlan->calibrated == false)
      m_execPlan = &m_defPlan;
#endif

   assert(m_execPlan != NULL && m_execPlan->calibrated);
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(numElements, output, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(numElements, output, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(numElements, output, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(numElements, output, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(numElements, output);
#endif
   case CPU_BACKEND:
      return CPU(numElements, output);

   default:
      return CPU(numElements, output);
   }
}





/*!
 *  Generates a number of elements for a matrix object using the user function which the skeleton was created with. The
 *  elements are saved in an output matrix.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param numRows The number of rows to be generated.
 *  \param numCols The number of columns to be generated.
 *  \param output The output matrix which will be overwritten with the generated values.
 */
template <typename GenerateFunc>
template <typename T>
void Generate<GenerateFunc>::operator()(size_t numRows, size_t numCols, Matrix<T>& output)
{
   if(m_generateFunc->funcType != GENERATE_MATRIX)
   {
      SKEPU_ERROR("Wrong operator type for Generate(Matrix)! Should use GENERATE_FUNC_MATRIX macro\n");
   }
   
   size_t size = numRows*numCols;
   

//      assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      m_execPlan = &m_execPlanMulti[0];
   }
   if (m_execPlan == NULL || m_execPlan->calibrated == false)
      m_execPlan = &m_defPlan;
#endif

   assert(m_execPlan != NULL && m_execPlan->calibrated);
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(numRows, numCols, output, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(numRows, numCols, output, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(numRows, numCols, output, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(numRows, numCols, output, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(numRows, numCols, output);
#endif
   case CPU_BACKEND:
      return CPU(numRows, numCols, output);

   default:
      return CPU(numRows, numCols, output);
   }
}

/*!
 *  Generates a number of elements using the user function which the skeleton was created with. The
 *  elements are saved in an output range.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param numElements The number of elements to be generated.
 *  \param outputBegin An iterator pointing to the first element in the range which will be overwritten with generated values.
 */
template <typename GenerateFunc>
template <typename OutputIterator>
void Generate<GenerateFunc>::operator()(size_t numElements, OutputIterator outputBegin)
{
   if(m_generateFunc->funcType != GENERATE)
   {
      SKEPU_ERROR("Wrong operator type for Generate! Should use GENERATE_FUNC macro\n");
      return;
   }
   
   size_t size = numElements;
   
   // for iterators, we don't apply any logic
   if (m_execPlan == NULL || m_execPlan->calibrated == false)
      m_execPlan = &m_defPlan;
   assert(m_execPlan != NULL && m_execPlan->calibrated);
   
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(numElements, outputBegin, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(numElements, outputBegin, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(numElements, outputBegin, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(numElements, outputBegin, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(numElements, outputBegin);
#endif
   case CPU_BACKEND:
      return CPU(numElements, outputBegin);

   default:
      return CPU(numElements, outputBegin);
   }
}

}

