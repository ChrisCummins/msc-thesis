/*! \file mapreduce.inl
 *  \brief Contains the definitions of non-backend specific member functions for the MapReduce skeleton.
 */

#ifdef SKEPU_OPENMP
#include <omp.h>
#endif

namespace skepu
{

/*!
 *  When creating an instance of the MapReduce skeleton, two pointers need to be provided. One for a
 *  unary, binary or trinary mapping function and one for a binary reduce function.
 *  Also the Environment is set and if \p SKEPU_OPENCL is defined, the appropriate OpenCL program and kernel are created.
 *  Also creates a default execution plan which the skeleton will use if no other is
 *  specified.
 *
 *  \param mapFunc A pointer to a valid unary, binary or trinary user function. Will be deleted in the destructor.
 *  \param reduceFunc A pointer to a valid binary user function. Will be deleted in the destructor.
 */
template <typename MapFunc, typename ReduceFunc>
MapReduce<MapFunc, ReduceFunc>::MapReduce(MapFunc* mapFunc, ReduceFunc* reduceFunc)
{
   BackEndParams bp;
   m_mapFunc = mapFunc;
   m_reduceFunc = reduceFunc;
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
   bp.maxThreads = m_environment->m_devices_CU.at(0)->getMaxThreads();
   bp.maxBlocks = m_environment->m_devices_CU.at(0)->getMaxBlocks();

   cudaDeviceID = Environment<int>::getInstance()->bestCUDADevID;
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
 *  When the MapReduce skeleton is destroyed, it deletes the user functions it was created with.
 */
template <typename MapFunc, typename ReduceFunc>
MapReduce<MapFunc, ReduceFunc>::~MapReduce()
{
   delete m_reduceFunc;
   delete m_mapFunc;
}

/*!
 *  Performs the Map on \em one Vector and Reduce on the result. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A vector which the map and reduce will be performed on.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::operator()(Vector<T>& input)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }
   if(m_reduceFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
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
 *  Performs the Map on \em one Matrix and Reduce on the result. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the map and reduce will be performed on.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::operator()(Matrix<T>& input)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }
   if(m_reduceFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
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
 *  Performs the Map on \em one range of elements and Reduce on the result. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param inputBegin An iterator to the first element in the range.
 *  \param inputEnd An iterator to the last element of the range.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename InputIterator>
typename InputIterator::value_type MapReduce<MapFunc, ReduceFunc>::operator()(InputIterator inputBegin, InputIterator inputEnd)
{
   if(m_mapFunc->funcType != UNARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }
   if(m_reduceFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }

   m_execPlan = &m_defPlan;
   assert(m_execPlan != NULL && m_execPlan->calibrated);

   int size = inputEnd - inputBegin;
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

/*!
 *  Performs the Map on \em two Vectors and Reduce on the result. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A Vector which the map and reduce will be performed on.
 *  \param input2 A Vector which the map and reduce will be performed on.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::operator()(Vector<T>& input1, Vector<T>& input2)
{
   if(m_mapFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }
   if(m_reduceFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }

   int n = input1.size();
   int n2 = input2.size();

   if( n != n2 )
   {
      std::cerr<<"Wrong input sizes!\n";
      return -1;
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

   BackEnd backEnd = m_execPlan->find(n);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input1, input2);
#endif
   case CPU_BACKEND:
      return CPU(input1, input2);

   default:
      return CPU(input1, input2);
   }
}


/*!
 *  Performs the Map on \em two matrices and Reduce on the result. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A matrix which the map and reduce will be performed on.
 *  \param input2 A matrix which the map and reduce will be performed on.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::operator()(Matrix<T>& input1, Matrix<T>& input2)
{
   if(m_mapFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }
   if(m_reduceFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }

   int n = input1.size();
   int n2 = input2.size();

   if( n != n2 )
   {
      SKEPU_ERROR("Wrong input sizes!\n");
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

   BackEnd backEnd = m_execPlan->find(n);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input1, input2);
#endif
   case CPU_BACKEND:
      return CPU(input1, input2);

   default:
      return CPU(input1, input2);
   }
}

/*!
 *  Performs the Map on \em two ranges of elements and Reduce on the result. Returns a scalar result.
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
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::operator()(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End)
{
   if(m_mapFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }
   if(m_reduceFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }

   int n = input1End - input1Begin;
   int n2 = input2End - input2Begin;

   if( n != n2 )
   {
      SKEPU_ERROR("Wrong input sizes!\n");
   }

   m_execPlan = &m_defPlan;
   BackEnd backEnd = m_execPlan->find(n);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1Begin, input1End, input2Begin, input2End, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1Begin, input1End, input2Begin, input2End, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1Begin, input1End, input2Begin, input2End, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1Begin, input1End, input2Begin, input2End, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input1Begin, input1End, input2Begin, input2End);
#endif
   case CPU_BACKEND:
      return CPU(input1Begin, input1End, input2Begin, input2End);

   default:
      return CPU(input1Begin, input1End, input2Begin, input2End);
   }
}

/*!
 *  Performs the Map on \em three Vectors and Reduce on the result. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A vector which the mapping will be performed on.
 *  \param input2 A vector which the mapping will be performed on.
 *  \param input3 A vector which the mapping will be performed on.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::operator()(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3)
{
   if(m_mapFunc->funcType != TERNARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }
   if(m_reduceFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }

   int n = input1.size();
   int n2 = input1.size();
   int n3 = input1.size();

   if( n != n2 ||  n != n3 )
   {
      SKEPU_ERROR("Wrong input sizes!\n");
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

   BackEnd backEnd = m_execPlan->find(n);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, input3, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, input3, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, input3, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, input3, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input1, input2, input3);
#endif
   case CPU_BACKEND:
      return CPU(input1, input2, input3);

   default:
      return CPU(input1, input2, input3);
   }
}


/*!
 *  Performs the Map on \em three matrices and Reduce on the result. Returns a scalar result.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A matrix which the mapping will be performed on.
 *  \param input2 A matrix which the mapping will be performed on.
 *  \param input3 A matrix which the mapping will be performed on.
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename T>
T MapReduce<MapFunc, ReduceFunc>::operator()(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3)
{
   if(m_mapFunc->funcType != TERNARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }
   if(m_reduceFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }

   int n = input1.size();
   int n2 = input1.size();
   int n3 = input1.size();

   if( n != n2 ||  n != n3 )
   {
      SKEPU_ERROR("Wrong input sizes!\n");
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

   BackEnd backEnd = m_execPlan->find(n);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, input3, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, input3, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, input3, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, input3, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input1, input2, input3);
#endif
   case CPU_BACKEND:
      return CPU(input1, input2, input3);

   default:
      return CPU(input1, input2, input3);
   }
}


/*!
 *  Performs the Map on \em three ranges of elements and Reduce on the result. Returns a scalar result.
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
 *  \return The scalar result of the map-reduce combination performed.
 */
template <typename MapFunc, typename ReduceFunc>
template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
typename Input1Iterator::value_type MapReduce<MapFunc, ReduceFunc>::operator()(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End)
{
   if(m_mapFunc->funcType != TERNARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }
   if(m_reduceFunc->funcType != BINARY)
   {
      SKEPU_ERROR("Wrong operator type!\n");
   }

   int n = input1End - input1Begin;
   int n2 = input2End - input2Begin;
   int n3 = input3End - input3Begin;

   if( n != n2 ||  n != n3 )
   {
      SKEPU_ERROR("Wrong input sizes!\n");
   }

   m_execPlan = &m_defPlan;
   BackEnd backEnd = m_execPlan->find(n);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End);
#endif
   case CPU_BACKEND:
      return CPU(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End);

   default:
      return CPU(input1Begin, input1End, input2Begin, input2End, input3Begin, input3End);
   }
}

}

