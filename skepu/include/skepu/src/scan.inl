/*! \file scan.inl
 *  \brief Contains the definitions of non-backend specific member functions for the Scan skeleton.
 */

#ifdef SKEPU_OPENMP
#include <omp.h>
#endif

namespace skepu
{

/*!
 *  When creating an instance of the Scan skeleton, a pointer to a binary user function must be provided.
 *  Also the environment is set and if \p SKEPU_OPENCL is defined, the appropriate OpenCL program
 *  and kernel are created. Also creates a default execution plan which the skeleton will use if no other is
 *  specified.
 *
 *  \param scanFunc A pointer to a valid binary user function. Will be deleted in the destructor.
 */
template <typename ScanFunc>
Scan<ScanFunc>::Scan(ScanFunc* scanFunc)
{
   if(scanFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type! Scan operation require binary user function.\n");
   }
   
   BackEndParams bp;
   m_scanFunc = scanFunc;
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
 *  When the Scan skeleton is destroyed, it deletes the user function it was created with.
 */
template <typename ScanFunc>
Scan<ScanFunc>::~Scan()
{
   delete  m_scanFunc;
}

/*!
 *  Performs the Scan on a whole Vector. With itself as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A vector which will be scanned. It will be overwritten with the result.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::operator()(Vector<T>& input, ScanType type, T init)
{
   int size = input.size();
   
//      assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      if(input.isVectorOnDevice_CU(cudaDeviceID))
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
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, type, init, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, type, init, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, type, init, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, type, init, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, type, init);
#endif
   case CPU_BACKEND:
      return CPU(input, type, init);

   default:
      return CPU(input, type, init);
   }
}



/*!
 *  Performs the Scan on a whole Matrix. With itself as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_OPENMP are defined then the OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which will be scanned. It will be overwritten with the result.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::operator()(Matrix<T>& input, ScanType type, T init)
{
#if defined(SKEPU_OPENMP)
   return OMP(input, type, init);
#else
   return CPU(input, type, init);
#endif
}

/*!
 *  Performs the Scan on a range of elements. With the same range as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename InputIterator>
void Scan<ScanFunc>::operator()(InputIterator inputBegin, InputIterator inputEnd, ScanType type, typename InputIterator::value_type init)
{
   int size = inputEnd - inputBegin;

   m_execPlan = &m_defPlan;
   assert(m_execPlan != NULL && m_execPlan->calibrated);
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, type, init, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, type, init, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, type, init, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, type, init, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(inputBegin, inputEnd, type, init);
#endif
   case CPU_BACKEND:
      return CPU(inputBegin, inputEnd, type, init);

   default:
      return CPU(inputBegin, inputEnd, type, init);
   }
}

/*!
 *  Performs the Scan on a whole Vector. With a seperate Vector as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A vector which will be scanned.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::operator()(Vector<T>& input, Vector<T>& output, ScanType type, T init)
{
   int size = input.size();

//      assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      if(input.isVectorOnDevice_CU(cudaDeviceID))
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
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output, type, init, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output, type, init, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, type, init, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, type, init, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, output, type, init);
#endif
   case CPU_BACKEND:
      return CPU(input, output, type, init);

   default:
      return CPU(input, output, type, init);
   }
}



/*!
 *  Performs the Scan on a whole Matrix. With a seperate Matrix as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_OPENMP are defined then the OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which will be scanned.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename T>
void Scan<ScanFunc>::operator()(Matrix<T>& input, Matrix<T>& output, ScanType type, T init)
{
#if defined(SKEPU_OPENMP)
   return OMP(input, output, type, init);
#else
   return CPU(input, output, type, init);
#endif
}



/*!
 *  Performs the Scan on a range of elements. With a seperate output range.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param type The scan type, either INCLUSIVE or EXCLUSIVE.
 *  \param init The initialization value for exclusive scans.
 */
template <typename ScanFunc>
template <typename InputIterator, typename OutputIterator>
void Scan<ScanFunc>::operator()(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init)
{
   int size = inputEnd - inputBegin;

   m_execPlan = &m_defPlan;
   assert(m_execPlan != NULL && m_execPlan->calibrated);
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, outputBegin, type, init, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, outputBegin, type, init, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, outputBegin, type, init, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, outputBegin, type, init, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(inputBegin, inputEnd, outputBegin, type, init);
#endif
   case CPU_BACKEND:
      return CPU(inputBegin, inputEnd, outputBegin, type, init);

   default:
      return CPU(inputBegin, inputEnd, outputBegin, type, init);
   }
}

}

