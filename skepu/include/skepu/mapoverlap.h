/*! \file mapoverlap.h
 *  \brief Contains a class declaration for the MapOverlap skeleton.
 */

#ifndef MAPOVERLAP_H
#define MAPOVERLAP_H

#ifdef SKEPU_OPENCL
#include <string>
#include <vector>
#ifdef USE_MAC_OPENCL
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "src/device_cl.h"
#endif

#include "src/environment.h"

#include "skepu/vector.h"
#include "skepu/matrix.h"

#include "src/operator_macros.h"
#include "src/exec_plan.h"

namespace skepu
{

/*!
 *  Enumeration of the different edge policies (what happens when a read outside the vector is performed) that the map overlap skeletons support.
 */



enum OverlapPolicy
{
   OVERLAP_ROW_WISE,
   OVERLAP_COL_WISE,
   OVERLAP_ROW_COL_WISE // ,  OVERLAP_NEIGHBOUR_WISE
};

enum EdgePolicy
{
   CONSTANT,
   CYCLIC,
   DUPLICATE
};

/*!
 *  \ingroup skeletons
 */

/*!
 *  \class MapOverlap
 *
 *  \brief A class representing the MapOverlap skeleton.
 *
 *  This class defines the MapOverlap skeleton which is similar to a Map, but each element of the result (vecor/matrix) is a function
 *  of \em several adjacent elements of one input (vecor/matrix) that reside at a certain constant maximum distance from each other.
 *  This class can be used to apply (1) overlap to a vector and (2) separable-overlap to a matrix (row-wise, column-wise). For
 *  non-separable matrix overlap which considers diagonal neighbours as well besides row- and column-wise neighbours, please see \p src/MapOverlap2D.
 *  MapOverlap2D class can be used by including same header file (i.e., mapoverlap.h) but class name is different (MapOverlap2D).
 *
 *  Once instantiated, it is meant to be used as a function and therefore overloading
 *  \p operator(). There are a few overloaded variants of this operator depending on if a seperate output vector is provided
 *  or if vectors or iterators are used as parameters.
 *
 *  If a certain backend is to be used, functions with the same interface as \p operator() but by the names \p CPU, \p CU,
 *  \p CL, \p OMP exists for CPU, CUDA, OpenCL and OpenMP respectively.
 *
 *  The MapOverlap skeleton also includes a pointer to an Environment which includes the devices available to execute on.
 */
template <typename MapOverlapFunc>
class MapOverlap
{

public:

   MapOverlap(MapOverlapFunc* mapOverlapFunc);

   ~MapOverlap();

   void finishAll()
   {
      m_environment->finishAll();
   }

   void setExecPlan(ExecPlan& plan)
   {
      m_execPlan = &plan;
   }
   void setExecPlan(ExecPlan *plan)
   {
      m_execPlanMulti = plan;
   }

private:
   Environment<int>* m_environment;
   MapOverlapFunc* m_mapOverlapFunc;

   /*! this is the pointer to array of execution plans in multiExecPlan scenario. Only one of them should be activated at a time considering current data locality */
   ExecPlan *m_execPlanMulti;

   /*! this is the pointer to execution plan that is active and should be used by implementations to check numOmpThreads and cudaBlocks etc. */
   ExecPlan *m_execPlan;

   /*! this is the default execution plan that gets created on initialization */
   ExecPlan m_defPlan;

public:
   template <typename T>
   void operator()(Vector<T>& input, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator>
   void operator()(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename T>
   void operator()(Matrix<T>& input, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator>
   void operator()(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename T>
   void operator()(Vector<T>& input, Vector<T>& output, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator, typename OutputIterator>
   void operator()(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename T>
   void operator()(Matrix<T>& input, Matrix<T>& output, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator, typename OutputIterator>
   void operator()(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, typename InputIterator::value_type pad = typename InputIterator::value_type());

public:
   template <typename T>
   void CPU(Vector<T>& input, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator>
   void CPU(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename T>
   void CPU(Matrix<T>& input, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator>
   void CPU(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly,  typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename T>
   void CPU(Vector<T>& input, Vector<T>& output, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator, typename OutputIterator>
   void CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename T>
   void CPU(Matrix<T>& input, Matrix<T>& output, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator, typename OutputIterator>
   void CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type());

private:
   template <typename InputIterator, typename OutputIterator>
   void CPU_ROWWISE(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly = CONSTANT, typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename InputIterator, typename OutputIterator>
   void CPU_COLWISE(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly = CONSTANT, typename InputIterator::value_type pad = typename InputIterator::value_type());

#ifdef SKEPU_OPENMP
public:
   template <typename T>
   void OMP(Vector<T>& input, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator>
   void OMP(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename T>
   void OMP(Matrix<T>& input, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator>
   void OMP(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename T>
   void OMP(Vector<T>& input, Vector<T>& output, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator, typename OutputIterator>
   void OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename T>
   void OMP(Matrix<T>& input, Matrix<T>& output, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, T pad = T());

   template <typename InputIterator, typename OutputIterator>
   void OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type());

private:
   template <typename InputIterator, typename OutputIterator>
   void OMP_ROWWISE(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly = CONSTANT, typename InputIterator::value_type pad = typename InputIterator::value_type());

   template <typename InputIterator, typename OutputIterator>
   void OMP_COLWISE(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly = CONSTANT, typename InputIterator::value_type pad = typename InputIterator::value_type());

#endif

#ifdef SKEPU_CUDA
public:
   template <typename T>
   void CU(Vector<T>& input, EdgePolicy poly = CONSTANT, T pad = T(), int useNumGPU = 1);

   template <typename InputIterator>
   void CU(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type(), int useNumGPU = 1);

   template <typename T>
   void CU(Matrix<T>& input, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, T pad = T(), int useNumGPU = 1);

   template <typename InputIterator>
   void CU(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type(), int useNumGPU = 1);

   template <typename T>
   void CU(Vector<T>& input, Vector<T>& output, EdgePolicy poly = CONSTANT, T pad = T(), int useNumGPU = 1);

   template <typename InputIterator, typename OutputIterator>
   void CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type(), int useNumGPU = 1);

   template <typename T>
   void CU(Matrix<T>& input, Matrix<T>& output, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, T pad = T(), int useNumGPU = 1);

   template <typename InputIterator, typename OutputIterator>
   void CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type(), int useNumGPU = 1);

private:
   unsigned int cudaDeviceID;

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID, OverlapPolicy overlapPolicy);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapSingleThread_CU_Row(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapMultiThread_CU_Row(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapSingleThread_CU_Col(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapMultiThread_CU_Col(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices);

   template <typename T>
   size_t getThreadNumber_CU(size_t width, size_t &numThreads, unsigned int deviceID);

   template <typename T>
   bool sharedMemAvailable_CU(size_t &numThreads, unsigned int deviceID);

#endif

#ifdef SKEPU_OPENCL
public:
   template <typename T>
   void CL(Vector<T>& input, EdgePolicy poly = CONSTANT, T pad = T(), int useNumGPU = 1);

   template <typename InputIterator>
   void CL(InputIterator inputBegin, InputIterator inputEnd, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type(), int useNumGPU = 1);

   template <typename T>
   void CL(Matrix<T>& input, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, T pad = T(), int useNumGPU = 1);

   template <typename InputIterator>
   void CL(InputIterator inputBegin, InputIterator inputEnd, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type(), int useNumGPU = 1);

   template <typename T>
   void CL(Vector<T>& input, Vector<T>& output, EdgePolicy poly = CONSTANT, T pad = T(), int useNumGPU = 1);

   template <typename InputIterator, typename OutputIterator>
   void CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type(), int useNumGPU = 1);

   template <typename T>
   void CL(Matrix<T>& input, Matrix<T>& output, OverlapPolicy overlapPolicy, EdgePolicy poly = CONSTANT, T pad = T(), int useNumGPU = 1);

   template <typename InputIterator, typename OutputIterator>
   void CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, OverlapPolicy overlapPolicy, EdgePolicy poly, typename InputIterator::value_type pad = typename InputIterator::value_type(), int useNumGPU = 1);

private:
   template <typename InputIterator, typename OutputIterator>
   void mapOverlapSingle_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapSingle_CL_Row(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapSingle_CL_RowMulti(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapSingle_CL_Col(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapSingle_CL_ColMulti(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapSingle_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, unsigned int deviceID, OverlapPolicy overlapPolicy);

   template <typename InputIterator, typename OutputIterator>
   void mapOverlapNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, EdgePolicy poly, typename InputIterator::value_type pad, size_t numDevices, OverlapPolicy overlapPolicy);

   template <typename T>
   size_t getThreadNumber_CL(size_t width, size_t &numThreads, unsigned int deviceID);

   template <typename T>
   bool sharedMemAvailable_CL(size_t &numThreads, unsigned int deviceID);

private:
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_CL;
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_Mat_Row_CL;
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_Mat_Col_CL;
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_Mat_ColMulti_CL;
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_Mat_ConvolFilter_CL;
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_Mat_Convol_CL;

   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_Mat_Transpose_CL;

   void createOpenCLProgram();
#endif

};





}

#include "src/mapoverlap.inl"

#include "src/mapoverlap_cpu.inl"

#ifdef SKEPU_OPENMP
#include "src/mapoverlap_omp.inl"
#endif

#ifdef SKEPU_OPENCL
#include "src/mapoverlap_cl.inl"
#endif

#ifdef SKEPU_CUDA
#include "src/mapoverlap_cu.inl"
#endif




//---------------------------------------------------------------------------------------------
//------------------------------------------------------- Adding MapOverlap2D type definitions
//---------------------------------------------------------------------------------------------

#include "src/2dmapoverlap.h"



#endif

