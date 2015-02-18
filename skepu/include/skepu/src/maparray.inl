/*! \file maparray.inl
 *  \brief Contains the definitions of non-backend specific member functions for the MapArray skeleton.
 */

#ifdef SKEPU_OPENMP
#include <omp.h>
#endif

namespace skepu
{

/*!
 *  When creating an instance of the MapArray skeleton, a pointer to an array user function must be provided.
 *  Also the environment is set and if \p SKEPU_OPENCL is defined, the appropriate OpenCL program
 *  and kernel are created. Also creates a default execution plan which the skeleton will use if no other is
 *  specified.
 *
 *  \param mapArrayFunc A pointer to a valid array user function. Will be deleted in the destructor.
 */
template <typename MapArrayFunc>
MapArray<MapArrayFunc>::MapArray(MapArrayFunc* mapArrayFunc)
{
   BackEndParams bp;
   m_mapArrayFunc = mapArrayFunc;
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
 *  When the MapArray skeleton is destroyed, it deletes the user function it was created with.
 */
template <typename MapArrayFunc>
MapArray<MapArrayFunc>::~MapArray()
{
   delete  m_mapArrayFunc;
}

/*!
 *  Performs the MapArray on the two Vectors specified. A seperate output vector is used.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 First vector, which can be accessed entirely for each element in second vector.
 *  \param input2 Second vector, each value of this vector can be mapped to several values from first Vector.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::operator()(Vector<T>& input1, Vector<T>& input2, Vector<T>& output, int groupId)
{
   if(m_mapArrayFunc->funcType != ARRAY)
   {
      SKEPU_ERROR("Wrong operator type for MapArray! Should use ARRAY_FUNC or ARRAY_FUNC_CONSTANT macro\n");
   }
   
   int size = input1.size();

// 	assert(/*m_execPlanMulti != NULL && */input1.size() == input2.size());
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL && input1.size() == input2.size())
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
      return CU(input1, input2, output, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, output, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, output, SKEPU_NUMGPU);
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
 *  Performs the MapArray on the one vector and matrix specified. A seperate output matrix is used.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 First vector, which can be accessed entirely for each element in second vector.
 *  \param input2 Second matrix, each value of this matrix can be mapped to several values from first Vector.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::operator()(Vector<T>& input1, Matrix<T>& input2, Matrix<T>& output)
{
   if(m_mapArrayFunc->funcType != ARRAY_INDEX)
   {
      SKEPU_ERROR("Wrong operator type for MapArray(Vector, Matrix, Vector)! Should use ARRAY_FUNC_MATR or ARRAY_FUNC_MATR_CONSTANT macro\n");
   }
   
   int size = input1.size();
   BackEnd backEnd = m_execPlan->find(size);

// 	assert(m_execPlanMulti != NULL && input1.size() == input2.size());
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL && input1.size() == input2.size())
   {
      /*! both are on GPU... */
      if(input1.isVectorOnDevice_CU(cudaDeviceID) && input2.isMatrixOnDevice_CU(cudaDeviceID))
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
      else if(input1.isVectorOnDevice_CU(cudaDeviceID) || input2.isMatrixOnDevice_CU(cudaDeviceID))
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

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, output, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, output, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, output, SKEPU_NUMGPU);
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
 *  Performs the MapArray block-wise on the one vector and matrix. A seperate output vector is used.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A vector, which can be accessed entirely for each element in second vector.
 *  \param input2 A matrix, a set of values (blockLength specified in user function) of this matrix can be mapped to several values from first Vector.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::operator()(Vector<T>& input1, Matrix<T>& input2, Vector<T>& output)
{
   if(m_mapArrayFunc->funcType != ARRAY_INDEX_BLOCK_WISE)
   {
      SKEPU_ERROR("Wrong operator type for MapArray(Vector, Matrix, Vector)! Should use ARRAY_FUNC_MATR_BLOCK_WISE macro\n");
   }
   
// 	assert(m_execPlanMulti != NULL && input1.size() == input2.size());
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL && input1.size() == input2.size())
   {
      /*! both are on GPU... */
      if(input1.isVectorOnDevice_CU(cudaDeviceID) && input2.isMatrixOnDevice_CU(cudaDeviceID))
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
      else if(input1.isVectorOnDevice_CU(cudaDeviceID) || input2.isMatrixOnDevice_CU(cudaDeviceID))
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

   int size = input1.size();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, output, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, output, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, output, SKEPU_NUMGPU);
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
 *  Performs the MapArray block-wise on the one vector and a sparse matrix. A seperate output vector is used.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1 A vector, which can be accessed entirely for each element in second vector.
 *  \param input2 A sparse matrix, a set of values (blockLength specified in user function) of this matrix can be mapped to several values from first Vector.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 */
template <typename MapArrayFunc>
template <typename T>
void MapArray<MapArrayFunc>::operator()(Vector<T>& input1, SparseMatrix<T>& input2, Vector<T>& output)
{
   if(m_mapArrayFunc->funcType != ARRAY_INDEX_SPARSE_BLOCK_WISE)
   {
      SKEPU_ERROR("Wrong operator type for MapArray(Vector, SparseMatrix, Vector)! Should use ARRAY_FUNC_SPARSE_MATR_BLOCK_WISE macro\n");
   }
   
// 	assert(m_execPlanMulti != NULL && input1.size() == input2.size());
#ifdef SKEPU_CUDA
   if(m_execPlanMulti != NULL && input1.size() == input2.size())
   {
      /*! both are on GPU... */
      if(input1.isVectorOnDevice_CU(cudaDeviceID) && input2.isSparseMatrixOnDevice_CU(cudaDeviceID))
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
      else if(input1.isVectorOnDevice_CU(cudaDeviceID) || input2.isSparseMatrixOnDevice_CU(cudaDeviceID))
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

   int size = input1.size();
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, output, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1, input2, output, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1, input2, output, SKEPU_NUMGPU);
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
 *  Performs the MapArray on the two element ranges specified. A seperate output range is used.
 *  First range can be accessed entirely for each element in second range.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input1Begin A Vector::iterator to the first element in the first range.
 *  \param input1End A Vector::iterator to the last element of the first range.
 *  \param input2Begin A Vector::iterator to the first element in the second range.
 *  \param input2End A Vector::iterator to the last element of the second range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 */
template <typename MapArrayFunc>
template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
void MapArray<MapArrayFunc>::operator()(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin)
{
   if(m_mapArrayFunc->funcType != ARRAY)
   {
      SKEPU_ERROR("Wrong operator type for MapArray! Should use ARRAY_FUNC or ARRAY_FUNC_CONSTANT macro\n");
   }
   
   int size = input1End - input1Begin;
   
   m_execPlan = &m_defPlan;
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1Begin, input1End, input2Begin, input2End, outputBegin, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input1Begin, input1End, input2Begin, input2End, outputBegin, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input1Begin, input1End, input2Begin, input2End, outputBegin, SKEPU_NUMGPU);
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



}

