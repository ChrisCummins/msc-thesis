/*! \file mapreduce.h
 *  \brief Contains a class declaration for the MapReduce skeleton.
 */

#ifndef MAPREDUCE_H
#define MAPREDUCE_H

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
#include "skepu/sparse_matrix.h"

#include "src/operator_macros.h"
#include "src/exec_plan.h"

namespace skepu
{

/*!
 *  \ingroup skeletons
 */

/*!
 *  \class MapReduce
 *
 *  \brief A class representing the MapReduce skeleton.
 *
 *  This class defines the MapReduce skeleton which is a combination of the Map and Reduce operations. It produces
 *  the same result as if one would first Map one or more vectors (matrices) to a result vector (matrix), then do a reduction on that result.
 *  It is provided since it combines the mapping and reduction in the same computation kernel and therefore avoids some
 *  synchronization, which speeds up the calculation. Once instantiated, it is meant to be used as a function and therefore overloading
 *  \p operator(). There are several overloaded versions of this operator that can be used depending on how many elements
 *  the mapping function uses (one, two or three). There are also variants which takes iterators as inputs and those that
 *  takes whole containers (vectors, matrices).
 *
 *  If a certain backend is to be used, functions with the same interface as \p operator() but by the names \p CPU, \p CU,
 *  \p CL, \p OMP exists for CPU, CUDA, OpenCL and OpenMP respectively.
 *
 *  The MapReduce skeleton also includes a pointer to an Environment which includes the devices available to execute on.
 */
template <typename MapFunc, typename ReduceFunc>
class MapReduce
{

public:

   MapReduce(MapFunc* mapFunc, ReduceFunc* reduceFunc);

   ~MapReduce();

   template <typename T>
   void setConstant(T constant1)
   {
      m_mapFunc->setConstant(constant1);
   }

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
   MapFunc* m_mapFunc;
   ReduceFunc* m_reduceFunc;

   /*! this is the pointer to array of execution plans in multiExecPlan scenario. Only one of them should be activated at a time considering current data locality */
   ExecPlan *m_execPlanMulti;

   /*! this is the pointer to execution plan that is active and should be used by implementations to check numOmpThreads and cudaBlocks etc. */
   ExecPlan *m_execPlan;

   /*! this is the default execution plan that gets created on initialization */
   ExecPlan m_defPlan;

public:
   template <typename T>
   T operator()(Vector<T>& input);

   template <typename T>
   T operator()(Matrix<T>& input);

   template <typename InputIterator>
   typename InputIterator::value_type operator()(InputIterator inputBegin, InputIterator inputEnd);

   template <typename T>
   T operator()(Vector<T>& input1, Vector<T>& input2);

   template <typename T>
   T operator()(Matrix<T>& input1, Matrix<T>& input2);

   template <typename Input1Iterator, typename Input2Iterator>
   typename Input1Iterator::value_type operator()(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End);

   template <typename T>
   T operator()(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3);

   template <typename T>
   T operator()(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
   typename Input1Iterator::value_type operator()(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End);

public:
   template <typename T>
   T CPU(Vector<T>& input);

   template <typename T>
   T CPU(Matrix<T>& input);

   template <typename InputIterator>
   typename InputIterator::value_type CPU(InputIterator inputBegin, InputIterator inputEnd);

   template <typename T>
   T CPU(Vector<T>& input1, Vector<T>& input2);

   template <typename T>
   T CPU(Matrix<T>& input1, Matrix<T>& input2);

   template <typename Input1Iterator, typename Input2Iterator>
   typename Input1Iterator::value_type CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End);

   template <typename T>
   T CPU(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3);

   template <typename T>
   T CPU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
   typename Input1Iterator::value_type CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End);

#ifdef SKEPU_OPENMP
public:
   template <typename T>
   T OMP(Vector<T>& input);

   template <typename T>
   T OMP(Matrix<T>& input);

   template <typename InputIterator>
   typename InputIterator::value_type OMP(InputIterator inputBegin, InputIterator inputEnd);

   template <typename T>
   T OMP(Vector<T>& input1, Vector<T>& input2);

   template <typename T>
   T OMP(Matrix<T>& input1, Matrix<T>& input2);

   template <typename Input1Iterator, typename Input2Iterator>
   typename Input1Iterator::value_type OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End);

   template <typename T>
   T OMP(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3);

   template <typename T>
   T OMP(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
   typename Input1Iterator::value_type OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End);
#endif

#ifdef SKEPU_CUDA
public:
   template <typename T>
   T CU(Vector<T>& input, int useNumGPU = 1);

   template <typename T>
   T CU(Matrix<T>& input, int useNumGPU = 1);

   template <typename InputIterator>
   typename InputIterator::value_type CU(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU = 1);

   template <typename T>
   T CU(Vector<T>& input1, Vector<T>& input2, int useNumGPU = 1);

   template <typename T>
   T CU(Matrix<T>& input1, Matrix<T>& input2, int useNumGPU = 1);

   template <typename Input1Iterator, typename Input2Iterator>
   typename Input1Iterator::value_type CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, int useNumGPU = 1);

   template <typename T>
   T CU(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, int useNumGPU = 1);

   template <typename T>
   T CU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, int useNumGPU = 1);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
   typename Input1Iterator::value_type CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, int useNumGPU = 1);

private:
   unsigned int cudaDeviceID;

   template <typename InputIterator>
   typename InputIterator::value_type mapReduceSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID);

   template <typename Input1Iterator, typename Input2Iterator>
   typename Input1Iterator::value_type mapReduceSingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, unsigned int deviceID);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
   typename Input1Iterator::value_type mapReduceSingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, unsigned int deviceID);

#endif

#ifdef SKEPU_OPENCL
public:
   template <typename T>
   T CL(Vector<T>& input, int useNumGPU = 1);

   template <typename T>
   T CL(Matrix<T>& input, int useNumGPU = 1);

   template <typename InputIterator>
   typename InputIterator::value_type CL(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU = 1);

   template <typename T>
   T CL(Vector<T>& input1, Vector<T>& input2, int useNumGPU = 1);

   template <typename T>
   T CL(Matrix<T>& input1, Matrix<T>& input2, int useNumGPU = 1);

   template <typename Input1Iterator, typename Input2Iterator>
   typename Input1Iterator::value_type CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, int useNumGPU = 1);

   template <typename T>
   T CL(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, int useNumGPU = 1);

   template <typename T>
   T CL(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, int useNumGPU = 1);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
   typename Input1Iterator::value_type CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, int useNumGPU = 1);

private:
   template <typename InputIterator>
   typename InputIterator::value_type mapReduceSingle_CL(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID);

   template <typename InputIterator>
   typename InputIterator::value_type mapReduceNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, size_t numDevices);

   template <typename Input1Iterator, typename Input2Iterator>
   typename Input1Iterator::value_type mapReduceSingle_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, unsigned int deviceID);

   template <typename Input1Iterator, typename Input2Iterator>
   typename Input1Iterator::value_type mapReduceNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, size_t numDevices);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
   typename Input1Iterator::value_type mapReduceSingle_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, unsigned int deviceID);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator>
   typename Input1Iterator::value_type mapReduceNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, size_t numDevices);

private:
   std::vector<std::pair<cl_kernel, Device_CL*> > m_mapReduceKernels_CL;
   std::vector<std::pair<cl_kernel, Device_CL*> > m_reduceKernels_CL;

   void createOpenCLProgram();
#endif

};




}

#include "src/mapreduce.inl"

#include "src/mapreduce_cpu.inl"

#ifdef SKEPU_OPENMP
#include "src/mapreduce_omp.inl"
#endif

#ifdef SKEPU_OPENCL
#include "src/mapreduce_cl.inl"
#endif

#ifdef SKEPU_CUDA
#include "src/mapreduce_cu.inl"
#endif

#endif

