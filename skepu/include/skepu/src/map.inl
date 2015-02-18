/*! \file map.inl
 *  \brief Contains the definitions of non-backend specific member functions for the Map skeleton.
 */

#ifdef SKEPU_OPENMP
#include <omp.h>
#endif

namespace skepu
{

/*!
 *  When creating an instance of the Map skeleton, a pointer to a unary, binary or trinary user function must be provided.
 *  Also the environment is set and if \p SKEPU_OPENCL is defined, the appropriate OpenCL program
 *  and kernel are created.
 *
 *  \param mapFunc A pointer to a valid user function. Will be deleted in the destructor.
 */
template <typename MapFunc>
Map<MapFunc>::Map(MapFunc* mapFunc)
{
   BackEndParams bp;
   m_mapFunc = mapFunc;
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
 *  When the Map skeleton is destroyed, it deletes the user function it was created with.
 */
template <typename MapFunc>
Map<MapFunc>::~Map()
{
   delete m_mapFunc;
}

/*!
 *  Performs the Map on one vector. With itself as output. The Map skeleton needs to
 *  be created with a unary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(Vector<T>& input)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use UNARY_FUNC or UNARY_FUNC_CONSTANT macro\n");
   }

   if( input.empty() )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

// 	assert(m_execPlanMulti != NULL);
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

   size_t size = input.size();

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
 *  Performs the Map on one vector. With a second vector as output. The Map skeleton needs to
 *  be created with a unary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(Vector<T>& input, Vector<T>& output, int groupId)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use UNARY_FUNC or UNARY_FUNC_CONSTANT macro\n");
   }

   if( input.empty() )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.size());
   }

// 	assert(m_execPlanMulti != NULL);
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

   size_t size = input.size();

   BackEnd backEnd;
   if(groupId > 0)
   {
      if(m_environment->getGroupMapping(groupId, backEnd) == false)
      {
         backEnd = m_execPlan->find(size);
         m_environment->addGroupMapping(groupId, backEnd);
      }
   }
   else
      backEnd = m_execPlan->find(size);

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
 *  Performs the Map on one matrix. With itself as output. The Map skeleton needs to
 *  be created with a unary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the mapping will be performed on. It will be overwritten with the result.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(Matrix<T>& input)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use UNARY_FUNC or UNARY_FUNC_CONSTANT macro\n");
   }

   if( input.empty() )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

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

   size_t size = input.size();
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
 *  Performs the Map on one matrix. With a second matrix as output. The Map skeleton needs to
 *  be created with a unary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(Matrix<T>& input, Matrix<T>& output)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use UNARY_FUNC or UNARY_FUNC_CONSTANT macro\n");
   }

   if( input.empty() )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

   if(input.size() != output.size())
   {
      output.clear();
      output.resize(input.total_rows(), input.total_cols());
   }

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

   size_t size = input.size();
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
 *  Performs the Map on a range of elements. The same range is used as output. The Map skeleton needs to
 *  be created with a unary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 */
template <typename MapFunc>
template <typename InputIterator>
void Map<MapFunc>::operator()(InputIterator inputBegin, InputIterator inputEnd)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use UNARY_FUNC or UNARY_FUNC_CONSTANT macro\n");
   }

   size_t size = inputEnd - inputBegin;

   if( size == 0 )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

   m_execPlan = &m_defPlan; // for iterators, we do not implement any logic yet...
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, 0);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, 0);
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

/*!
 *  Performs the Map on a range of elements. A seperate output range is used. The Map skeleton needs to
 *  be created with a unary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \param outputBegin An iterator to the first element of the output range.
 */
template <typename MapFunc>
template <typename InputIterator, typename OutputIterator>
void Map<MapFunc>::operator()(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use UNARY_FUNC or UNARY_FUNC_CONSTANT macro\n");
   }

   size_t size = inputEnd - inputBegin;

   if( size == 0 )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

   m_execPlan = &m_defPlan; // for iterators, we do not implement any logic yet...
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, outputBegin, 0);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, outputBegin, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, outputBegin, 0);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, outputBegin, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(inputBegin, inputEnd, outputBegin);
#endif
   case CPU_BACKEND:
      return CPU(inputBegin, inputEnd, outputBegin);

   default:
      return CPU(inputBegin, inputEnd, outputBegin);
   }
}

/*!
 *  Performs the Map on two vectors. With a seperate output. The Map skeleton needs to
 *  be created with a binary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(Vector<T>& input1, Vector<T>& input2, Vector<T>& output)
{
   if(m_mapFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use BINARY_FUNC or BINARY_FUNC_CONSTANT macro\n");
   }

   if( input1.empty() )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

   if( input1.size() != input2.size() )
   {
      SKEPU_ERROR("Input sizes mismatch in MAP: " << input1.size() <<", " << input2.size() <<"\n");
   }

   if(input1.size() != output.size())
   {
      output.clear();
      output.resize(input1.size());
   }

// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      /*! both are on GPU... */
      if(input1.isVectorOnDevice_CU(cudaDeviceID) && input2.isVectorOnDevice_CU(cudaDeviceID))
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
      else if(input1.isVectorOnDevice_CU(cudaDeviceID) || input2.isVectorOnDevice_CU(cudaDeviceID))
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

   assert(m_execPlan != NULL && m_execPlan->calibrated);

   size_t size = input1.size();
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
 *  Performs the Map on two matrices. With a seperate output. The Map skeleton needs to
 *  be created with a binary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& output)
{
   if(m_mapFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use BINARY_FUNC or BINARY_FUNC_CONSTANT macro\n");
   }

   if( input1.empty() )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

   if( input1.size() != input2.size() )
   {
      SKEPU_ERROR("Input sizes mismatch in MAP: " << input1.size() <<", " << input2.size() <<"\n");
   }

   if(input1.size() != output.size())
   {
      output.clear();
      output.resize(input1.total_rows(), input1.total_cols());
   }

// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      /*! both are on GPU... */
      if(input1.isMatrixOnDevice_CU(cudaDeviceID) && input2.isMatrixOnDevice_CU(cudaDeviceID))
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
      else if(input1.isMatrixOnDevice_CU(cudaDeviceID) || input2.isMatrixOnDevice_CU(cudaDeviceID))
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

   assert(m_execPlan != NULL && m_execPlan->calibrated);

   size_t size = input1.size();
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
 *  Performs the Map on two ranges of elements. With a seperate output range.
 *  The Map skeleton needs to be created with a binary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param outputBegin An iterator to the first element of the output range.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
void Map<MapFunc>::operator()(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin)
{
   if(m_mapFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use BINARY_FUNC or BINARY_FUNC_CONSTANT macro\n");
   }

   size_t size = input1End - input1Begin;

   if( size == 0 )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

   if( size != (input2End - input2Begin) )
   {
      SKEPU_ERROR("Input sizes mismatch in MAP: " << size <<", " <<(input2End - input2Begin) <<"\n");
   }

   m_execPlan = &m_defPlan; // for iterators, we do not implement any logic yet...
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1Begin, input1End, input2Begin, input2End, outputBegin, 0);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1Begin, input1End, input2Begin, input2End, outputBegin, 1);
#endif

   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1Begin, input1End, input2Begin, input2End, outputBegin, 0);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1Begin, input1End, input2Begin, input2End, outputBegin, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input1Begin, input1End, input2Begin, input2End, outputBegin);
#endif
   case CPU_BACKEND:
      return CPU(input1Begin, input1End, input2Begin, input2End, outputBegin);

   default:
      return CPU(input1Begin, input1End, input2Begin, input2End, outputBegin);
   }
}

/*!
 *  Performs the Map on three vectors. With a seperate output. The Map skeleton needs to
 *  be created with a trinary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param input3 A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, Vector<T>& output)
{
   if(m_mapFunc->funcType != TERNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use TERNARY_FUNC or TERNARY_FUNC_CONSTANT macro\n");
   }

   if( input1.empty() )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

   if( (input1.size() != input2.size()) || (input1.size() != input3.size()) )
   {
      SKEPU_ERROR("Input sizes mismatch in MAP: " << input1.size() <<", " << input2.size() <<", " << input3.size() <<"\n");
   }

   if(input1.size() != output.size())
   {
      output.clear();
      output.resize(input1.size());
   }

// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      /*! all three are on GPU... */
      if(input1.isVectorOnDevice_CU(cudaDeviceID) && input2.isVectorOnDevice_CU(cudaDeviceID) && input3.isVectorOnDevice_CU(cudaDeviceID))
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
      else if( (input1.isVectorOnDevice_CU(cudaDeviceID) && input2.isVectorOnDevice_CU(cudaDeviceID)) ||
               (input1.isVectorOnDevice_CU(cudaDeviceID) && input3.isVectorOnDevice_CU(cudaDeviceID)) ||
               (input2.isVectorOnDevice_CU(cudaDeviceID) && input3.isVectorOnDevice_CU(cudaDeviceID)) )
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
      else if (input1.isVectorOnDevice_CU(cudaDeviceID) || input2.isVectorOnDevice_CU(cudaDeviceID) || input3.isVectorOnDevice_CU(cudaDeviceID))
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

   assert(m_execPlan != NULL && m_execPlan->calibrated);

   size_t size = input1.size();
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



/*!
 *  Performs the Map on three matrices. With a seperate output. The Map skeleton needs to
 *  be created with a trinary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param input3 A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapFunc>
template <typename T>
void Map<MapFunc>::operator()(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, Matrix<T>& output)
{
   if(m_mapFunc->funcType != TERNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use TERNARY_FUNC or TERNARY_FUNC_CONSTANT macro\n");
   }

   if( input1.empty() )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

   if( (input1.size() != input2.size()) || (input1.size() != input3.size()) )
   {
      SKEPU_ERROR("Input sizes mismatch in MAP: " << input1.size() <<", " << input2.size() <<", " << input3.size() <<"\n");
   }

   if(input1.size() != output.size())
   {
      output.clear();
      output.resize(input1.total_rows(), input1.total_cols());
   }

// 	assert(m_execPlanMulti != NULL);
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL)
   {
      /*! all three are on GPU... */
      if(input1.isMatrixOnDevice_CU(cudaDeviceID) && input2.isMatrixOnDevice_CU(cudaDeviceID) && input3.isMatrixOnDevice_CU(cudaDeviceID))
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
      else if( (input1.isMatrixOnDevice_CU(cudaDeviceID) && input2.isMatrixOnDevice_CU(cudaDeviceID)) ||
               (input1.isMatrixOnDevice_CU(cudaDeviceID) && input3.isMatrixOnDevice_CU(cudaDeviceID)) ||
               (input2.isMatrixOnDevice_CU(cudaDeviceID) && input3.isMatrixOnDevice_CU(cudaDeviceID)) )
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
      else if (input1.isMatrixOnDevice_CU(cudaDeviceID) || input2.isMatrixOnDevice_CU(cudaDeviceID) || input3.isMatrixOnDevice_CU(cudaDeviceID))
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

   assert(m_execPlan != NULL && m_execPlan->calibrated);

   size_t size = input1.size();
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



/*!
 *  Performs the Map on three ranges of elements. With a seperate output range.
 *  The Map skeleton needs to be created with a trinary user function.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1Begin An iterator to the first element in the first range.
 *  \param input1End An iterator to the last element of the first range.
 *  \param input2Begin An iterator to the first element in the second range.
 *  \param input2End An iterator to the last element of the second range.
 *  \param input3Begin An iterator to the first element in the third range.
 *  \param input3End An iterator to the last element of the third range.
 *  \param outputBegin An iterator to the first element of the output range.
 */
template <typename MapFunc>
template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
void Map<MapFunc>::operator()(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin)
{
   if(m_mapFunc->funcType != TERNARY)
   {
      SKEPU_ERROR("Wrong operator type for Map! Should use TERNARY_FUNC or TERNARY_FUNC_CONSTANT macro\n");
   }

   size_t size = input1End - input1Begin;

   if( size == 0 )
   {
      SKEPU_ERROR("Map call: The input operand have no elements \n");
   }

   if( (size != (input2End - input2Begin)) || (size != (input3End - input3Begin)) )
   {
      SKEPU_ERROR("Input sizes mismatch in MAP: " << size <<", " <<(input2End - input2Begin)<<", " <<(input3End - input3Begin) <<"\n");
   }

   m_execPlan = &m_defPlan; // for iterators, we do not implement any logic yet...
   
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, outputBegin, 0);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, outputBegin, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, outputBegin, 0);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, outputBegin, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, outputBegin);
#endif
   case CPU_BACKEND:
      return CPU(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, outputBegin);

   default:
      return CPU(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, outputBegin);
   }
}


}



