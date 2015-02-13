/*! \file 2dmapoverlap.h
 *  \brief Contains a class declaration for the MapOverlap skeleton.
 */

#ifndef MAPOVERLAP_2D_H
#define MAPOVERLAP_2D_H

#ifdef SKEPU_OPENCL
#include <string>
#include <vector>
#ifdef USE_MAC_OPENCL
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "device_cl.h"
#endif

#include "environment.h"
#include "../matrix.h"
#include "operator_macros.h"
#include "exec_plan.h"

namespace skepu
{


/*!
 *  \ingroup skeletons
 */

/*!
 *  \class MapOverlap2D
 *
 *  \brief A class representing the MapOverlap skeleton for 2D overlap for Matrix operands (useful for convolution and stencil computation).
 *
 *  This class defines the MapOverlap skeleton for 2-dimensional MapOverlap for Matrix operands which is
 *  similar to a normal MapOverlap, but each element of the result matrix is a function of \em several
 *  "neighboring" elements of one input matrix that reside at a certain constant maximum distance, control
 *  by overlap-width and overlap-height. Once instantiated, it is meant to be used as a function and therefore
 *  overloading \p operator(). There are a few overloaded variants of this operator depending on whenther
 *  (1) user-function is used to specify the neighborhood filter or (2) a filter matrix is passed instead, specifiying
 *  weights for each neighbouring elements. Later is optimized when using CUDA backend using tiling optimization.
 *
 *  If a certain backend is to be used, functions with the same interface as \p operator() but by the names \p CPU, \p CU,
 *  \p CL, \p OMP exists for CPU, CUDA, OpenCL and OpenMP respectively.
 *
 *  The MapOverlap2D skeleton also includes a pointer to an Environment which includes the devices available to execute on.
 */
template <typename MapOverlap2DFunc>
class MapOverlap2D
{

public:

   MapOverlap2D(MapOverlap2DFunc* mapOverlapFunc);

   ~MapOverlap2D();

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
   MapOverlap2DFunc* m_mapOverlapFunc;

   /*! this is the pointer to array of execution plans in multiExecPlan scenario. Only one of them should be activated at a time considering current data locality */
   ExecPlan *m_execPlanMulti;

   /*! this is the pointer to execution plan that is active and should be used by implementations to check numOmpThreads and cudaBlocks etc. */
   ExecPlan *m_execPlan;

   /*! this is the default execution plan that gets created on initialization */
   ExecPlan m_defPlan;

public:
   template <typename T>
   void operator()(Matrix<T>& input);

   template <typename T>
   void operator()(Matrix<T>& input, Matrix<T>& output);

   template <typename T>
   void operator()(Matrix<T>& input, Matrix<T>& output, Matrix<T>& filter, bool useTiling=false);

   template <typename T>
   void operator()(Matrix<T>& input, Matrix<T>& output, size_t filter_rows, size_t filter_cols, bool useTiling=false);

public:
   template <typename T>
   void CPU(Matrix<T>& input);

   template <typename T>
   void CPU(Matrix<T>& input, Matrix<T>& output);

   template <typename T>
   void CPU(Matrix<T>& input, Matrix<T>& output, Matrix<T>& filter);

   template <typename T>
   void CPU(Matrix<T>& input, Matrix<T>& output, size_t filter_rows, size_t filter_cols);

#ifdef SKEPU_OPENMP
public:
   template <typename T>
   void OMP(Matrix<T>& input);

   template <typename T>
   void OMP(Matrix<T>& input, Matrix<T>& output);

   template <typename T>
   void OMP(Matrix<T>& input, Matrix<T>& output, Matrix<T>& filter);

   template <typename T>
   void OMP(Matrix<T>& input, Matrix<T>& output, size_t filter_rows, size_t filter_cols);
#endif

#ifdef SKEPU_CUDA
public:
   template <typename T>
   void CU(Matrix<T>& input, int useNumGPU = 1);

   template <typename T>
   void CU(Matrix<T>& input, Matrix<T>& output, int useNumGPU = 1);

   template <typename T>
   void CU(Matrix<T>& input, Matrix<T>& output, Matrix<T>& filter, bool useTiling=false, int useNumGPU=1);

   template <typename T>
   void CU(Matrix<T>& input, Matrix<T>& output, size_t filter_rows, size_t filter_cols, bool useTiling=false, int useNumGPU=1);

private:
   unsigned int cudaDeviceID;
   
   template <typename T>
   void mapOverlapSingleThread_CU(Matrix<T>& input, Matrix<T>& output, unsigned int deviceID);

   template <typename T>
   void mapOverlapMultipleThread_CU(Matrix<T>& input, Matrix<T>& output, size_t numDevices);
#endif

#ifdef SKEPU_OPENCL
public:
   template <typename T>
   void CL(Matrix<T>& input, int useNumGPU = 1);

   template <typename T>
   void CL(Matrix<T>& input, Matrix<T>& output, int useNumGPU = 1);

   template <typename T>
   void CL(Matrix<T>& input, Matrix<T>& output, Matrix<T>& filter, int useNumGPU = 1);

   template <typename T>
   void CL(Matrix<T>& input, Matrix<T>& output, size_t filter_rows, size_t filter_cols, int useNumGPU = 1);

private:
   template <typename T>
   void mapOverlapSingleThread_CL(Matrix<T>& input, Matrix<T>& output, unsigned int deviceID);

   template <typename T>
   void mapOverlapMultipleThread_CL(Matrix<T>& input, Matrix<T>& output, size_t numDevices);

private:
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_2D_CL;

   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_Mat_ConvolFilter_CL;
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_Mat_Convol_CL;

   void replaceText(std::string& text, std::string find, std::string replace);
   void createOpenCLProgram();
#endif

};





}

#include "2dmapoverlap.inl"

#include "2dmapoverlap_cpu.inl"

#ifdef SKEPU_OPENMP
#include "2dmapoverlap_omp.inl"
#endif

#ifdef SKEPU_OPENCL
#include "2dmapoverlap_cl.inl"
#endif

#ifdef SKEPU_CUDA
#include "2dmapoverlap_cu.inl"
#endif

#endif

