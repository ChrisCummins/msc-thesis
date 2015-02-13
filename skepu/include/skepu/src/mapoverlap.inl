/*! \file mapoverlap.inl
 *  \brief Contains the definitions of non-backend specific member functions for the MapOverlap skeleton.
 */

#ifdef SKEPU_OPENMP
#include <omp.h>
#endif


#define BLOCK_DIM 16


namespace skepu
{

/*!
 *  When creating an instance of the MapOverlap skeleton, a pointer to an overlap user function must be provided.
 *  Also the environment is set and if \p SKEPU_OPENCL is defined, the appropriate OpenCL program
 *  and kernel are created. Also creates a default execution plan which the skeleton will use if no other is
 *  specified.
 *
 *  \param mapOverlapFunc A pointer to a valid overlap user function. Will be deleted in the destructor.
 */
template <typename MapOverlapFunc>
MapOverlap<MapOverlapFunc>::MapOverlap(MapOverlapFunc* mapOverlapFunc)
{
   if(mapOverlapFunc->funcType != OVERLAP)
   {
      SKEPU_ERROR("Wrong operator type for MapOverlap skeleton!\n");
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
   m_execPlanMulti = NULL;

#ifdef SKEPU_OPENCL
   createOpenCLProgram();
#endif
}

/*!
 *  When the MapOverlap skeleton is destroyed, it deletes the user function it was created with.
 */
template <typename MapOverlapFunc>
MapOverlap<MapOverlapFunc>::~MapOverlap()
{
   delete  m_mapOverlapFunc;
}

/*!
 *  Performs the MapOverlap on a whole Vector. With itself as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A vector which the mapping will be performed on. It will be overwritten with the result.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::operator()(Vector<T>& input, EdgePolicy poly, T pad)
{
   size_t size = input.size();

//----- START: error check for possible erroneous overlap value ------//
   size_t overlap = m_mapOverlapFunc->overlap;
   if(overlap<1)
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap should be positive (grater than 0).\n");
   }
   else if (overlap>(size/2))
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap is too much. Should be atmost half of the size of the actual input.\n");
   }
//----- END: error check for possible erroneous overlap value ------//

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

   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, poly, pad, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, poly, pad, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, poly, pad, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, poly, pad, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, poly, pad);
#endif
   case CPU_BACKEND:
      return CPU(input, poly, pad);

   default:
      return CPU(input, poly, pad);
   }
}

/*!
 *  Performs the MapOverlap on a range of elements. With the same range as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename InputIterator>
void MapOverlap<MapOverlapFunc>::operator()(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad)
{
   size_t size = inputEnd - inputBegin;

//----- START: error check for possible erroneous overlap value ------//
   size_t overlap = m_mapOverlapFunc->overlap;
   if(overlap<1)
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap should be positive (grater than 0).\n");
   }
   else if (overlap>(size/2))
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap is too much. Should be atmost half of the size of the actual input.\n");
   }
//----- END: error check for possible erroneous overlap value ------//

   m_execPlan = &m_defPlan;
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, poly, pad, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, poly, pad, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, poly, pad, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, poly, pad, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(inputBegin, inputEnd, poly, pad);
#endif
   case CPU_BACKEND:
      return CPU(inputBegin, inputEnd, poly, pad);

   default:
      return CPU(inputBegin, inputEnd, poly, pad);
   }
}

/*!
 *  Performs the MapOverlap on a whole Matrix. With itself as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the mapping will be performed on. It will be overwritten with the result.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::operator()(Matrix<T>& input, OverlapPolicy overlapPolicy, EdgePolicy poly, T pad)
{
   size_t size = input.size();

//----- START: error check for possible erroneous overlap value ------//
   size_t overlap = m_mapOverlapFunc->overlap;

   // below condition actually gets the size (rows, cols) of matrix in which direction the overlap is applied.
   size_t comp_size = (   (overlapPolicy == OVERLAP_COL_WISE) ? input.total_rows() :
                       ( (overlapPolicy == OVERLAP_ROW_WISE) ? input.total_cols() :
                         ( (input.total_cols()>input.total_rows()) ? input.total_rows() : input.total_cols() ) )
                   );
   if(overlap<1)
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap should be positive (grater than 0).\n");
   }
   else if (overlap>(comp_size/2))
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap is too much. Should be atmost half of the size of the input matrix' dimension upon which overlap is applied.\n");
   }
//----- END: error check for possible erroneous overlap value ------//

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

   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, overlapPolicy, poly, pad, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, overlapPolicy, poly, pad, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, overlapPolicy, poly, pad, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, overlapPolicy, poly, pad, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, overlapPolicy, poly, pad);
#endif
   case CPU_BACKEND:
      return CPU(input, overlapPolicy, poly, pad);

   default:
      return CPU(input, overlapPolicy, poly, pad);
   }
}

/*!
 *  Performs the MapOverlap on a range of elements. With the same range as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 */
template <typename MapOverlapFunc>
template <typename InputIterator>
void MapOverlap<MapOverlapFunc>::operator()(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad)
{
   size_t size = inputEnd - inputBegin;

//----- START: error check for possible erroneous overlap value ------//
   size_t rows = inputBegin.getParent().total_rows();
   size_t cols = inputBegin.getParent().total_cols();

   if(size%cols==0)
   {
      SKEPU_ERROR("The MapOverlap on matrix can be applied only on a \"proper\" subset of Matrix.\n");
   }

   size_t overlap = m_mapOverlapFunc->overlap;

   // below condition actually gets the size (rows, cols) of matrix in which direction the overlap is applied.
   size_t comp_size = (   (overlapPolicy == OVERLAP_COL_WISE) ? rows :
                       ( (overlapPolicy == OVERLAP_ROW_WISE) ? cols :
                         ( (cols>rows) ? rows : cols ) )
                   );
   if(overlap<1)
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap should be positive (grater than 0).\n");
   }
   else if (overlap>(comp_size/2))
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap is too much. Should be atmost half of the size of the input matrix' dimension upon which overlap is applied.\n");
   }
//----- END: error check for possible erroneous overlap value ------//
   m_execPlan = &m_defPlan;
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, overlapPolicy, poly, pad, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, overlapPolicy, poly, pad, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, overlapPolicy, poly,pad, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, overlapPolicy, poly,pad, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(inputBegin, inputEnd, overlapPolicy, poly,pad);
#endif
   case CPU_BACKEND:
      return CPU(inputBegin, inputEnd, overlapPolicy, poly,pad);

   default:
      return CPU(inputBegin, inputEnd, overlapPolicy, poly,pad);
   }
}


/*!
 *  Performs the MapOverlap on a whole Vector. With a seperate Vector as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A vector which the mapping will be performed on.
 *  \param output The result vector, will be overwritten with the result and resized if needed.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::operator()(Vector<T>& input, Vector<T>& output, EdgePolicy poly, T pad)
{
   size_t size = input.size();

//----- START: error check for possible erroneous overlap value ------//
   size_t overlap = m_mapOverlapFunc->overlap;
   if(overlap<1)
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap should be positive (grater than 0).\n");

   }
   else if (overlap>(size/2))
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap is too much. Should be atmost half of the size of the actual input.\n");

   }
//----- END: error check for possible erroneous overlap value ------//

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

   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output, poly, pad, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output, poly, pad, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, poly, pad, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output, poly, pad, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, output, poly, pad);
#endif
   case CPU_BACKEND:
      return CPU(input, output, poly, pad);

   default:
      return CPU(input, output, poly, pad);
   }
}

/*!
 *  Performs the MapOverlap on a range of elements. With a seperate output range.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param inputBegin A Vector::iterator to the first element in the range.
 *  \param inputEnd A Vector::iterator to the last element of the range.
 *  \param outputBegin A Vector::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::operator()(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad)
{
   size_t size = inputEnd - inputBegin;

//----- START: error check for possible erroneous overlap value ------//
   size_t overlap = m_mapOverlapFunc->overlap;
   if(overlap<1)
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap should be positive (grater than 0).\n");
   }
   else if (overlap>(size/2))
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap is too much. Should be atmost half of the size of the actual input.\n");
   }
//----- END: error check for possible erroneous overlap value ------//

   m_execPlan = &m_defPlan;
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, outputBegin, poly, pad, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, outputBegin, poly, pad, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, outputBegin, poly, pad, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, outputBegin, poly, pad, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(inputBegin, inputEnd, outputBegin, poly, pad);
#endif
   case CPU_BACKEND:
      return CPU(inputBegin, inputEnd, outputBegin, poly, pad);

   default:
      return CPU(inputBegin, inputEnd, outputBegin, poly, pad);
   }
}


/*!
 *  Performs the MapOverlap on a whole Matrix. With a seperate Matrix as output.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 */
template <typename MapOverlapFunc>
template <typename T>
void MapOverlap<MapOverlapFunc>::operator()(Matrix<T>& input, Matrix<T>& output, OverlapPolicy overlapPolicy, EdgePolicy poly, T pad)
{
   size_t size = input.size();

//----- START: error check for possible erroneous overlap value ------//
   size_t overlap = m_mapOverlapFunc->overlap;

   // below condition actually gets the size (rows, cols) of matrix in which direction the overlap is applied.
   size_t comp_size = (   (overlapPolicy == OVERLAP_COL_WISE) ? input.total_rows() :
                       ( (overlapPolicy == OVERLAP_ROW_WISE) ? input.total_cols() :
                         ( (input.total_cols()>input.total_rows()) ? input.total_rows() : input.total_cols() ) )
                   );
   if(overlap<1)
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap should be positive (grater than 0).\n");
   }
   else if (overlap>(comp_size/2))
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap is too much. Should be atmost half of the size of the input matrix' dimension upon which overlap is applied.\n");
   }
//----- END: error check for possible erroneous overlap value ------//


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

   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output,overlapPolicy, poly, pad, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(input, output,overlapPolicy, poly, pad, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output,overlapPolicy, poly, pad, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(input, output,overlapPolicy, poly, pad, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(input, output,overlapPolicy, poly, pad);
#endif
   case CPU_BACKEND:
      return CPU(input, output,overlapPolicy, poly, pad);

   default:
      return CPU(input, output,overlapPolicy, poly, pad);
   }
}

/*!
 *  Performs the MapOverlap on a range of elements. With a seperate output range.
 *
 *  Depending on which backend was used, different versions of the skeleton is called.
 *  If \p SKEPU_CUDA is defined, the CUDA backend is used, similarly if \p SKEPU_OPENCL
 *  or \p SKEPU_OPENMP are defined then the OpenCL or OpenMP backend is used. As a fallback
 *  there is always a CPU version.
 *
 *  \param inputBegin A Matrix::iterator to the first element in the range.
 *  \param inputEnd A Matrix::iterator to the last element of the range.
 *  \param outputBegin A Matrix::iterator to the first element of the output range.
 *  \param poly Edge policy for the calculation, either CONSTANT or CYCLIC.
 *  \param pad If CONSTANT edge policy, this is the value to pad with.
 *  \param overlapPolicy Specify how to apply overlap either row-wise, column-wise or both.
 */
template <typename MapOverlapFunc>
template <typename InputIterator, typename OutputIterator>
void MapOverlap<MapOverlapFunc>::operator()(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad)
{
   size_t size = inputEnd - inputBegin;

//----- START: error check for possible erroneous overlap value ------//
   size_t rows = inputBegin.getParent().total_rows();
   size_t cols = inputBegin.getParent().total_cols();

   if(size%cols==0)
   {
      SKEPU_ERROR("The MapOverlap on matrix can be applied only on a \"proper\" subset of Matrix.\n");
   }

   size_t overlap = m_mapOverlapFunc->overlap;

   // below condition actually gets the size (rows, cols) of matrix in which direction the overlap is applied.
   size_t comp_size = (   (overlapPolicy == OVERLAP_COL_WISE) ? rows :
                       ( (overlapPolicy == OVERLAP_ROW_WISE) ? cols :
                         ( (cols>rows) ? rows : cols ) )
                   );
   if(overlap<1)
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap should be positive (grater than 0).\n");
   }
   else if (overlap>(comp_size/2))
   {
      SKEPU_ERROR("MapOverlap skeleton. Overlap is too much. Should be atmost half of the size of the input matrix' dimension upon which overlap is applied.\n");
   }
//----- END: error check for possible erroneous overlap value ------//

   m_execPlan = &m_defPlan;
   BackEnd backEnd = m_execPlan->find(size);

   switch(backEnd)
   {
   case CUM_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, outputBegin, overlapPolicy, poly, pad, SKEPU_NUMGPU);
#endif
   case CU_BACKEND:
#if defined(SKEPU_CUDA)
      return CU(inputBegin, inputEnd, outputBegin, overlapPolicy, poly, pad, 1);
#endif
   case CLM_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, outputBegin, overlapPolicy, poly, pad, SKEPU_NUMGPU);
#endif
   case CL_BACKEND:
#if defined(SKEPU_OPENCL)
      return CL(inputBegin, inputEnd, outputBegin, overlapPolicy, poly, pad, 1);
#endif
   case OMP_BACKEND:
#if defined(SKEPU_OPENMP)
      return OMP(inputBegin, inputEnd, outputBegin, overlapPolicy, poly, pad);
#endif
   case CPU_BACKEND:
      return CPU(inputBegin, inputEnd, outputBegin, overlapPolicy, poly, pad);

   default:
      return CPU(inputBegin, inputEnd, outputBegin, overlapPolicy, poly, pad);
   }
}

}

