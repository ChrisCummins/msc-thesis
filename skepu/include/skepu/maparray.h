/*! \file maparray.h
 *  \brief Contains a class declaration for the MapArray skeleton.
 */

#ifndef MAP_ARRAY_H
#define MAP_ARRAY_H

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
 *  \class MapArray
 *
 *  \brief A class representing the MapArray skeleton.
 *
 *  This class defines the MapArray skeleton. MapArray is yet another variant of Map.
 *  It produces a result (vector/matrix) from either two input objects (vector/matrix)
 *  where each element of the result, is a
 *   function of the corresponding element of the second input (vector/matrix),
 *  and any number of elements from the first input (vector). This means that at each call to the user defined function, which is
 *  done for each element in input two, all elements from input one can be accessed.
 *  Once instantiated, it is meant to be used as a function and therefore overloading
 *  \p operator(). There are a few overloaded variants of this operator. One using containers as inputs and the other using
 *  iterators to define a range of elements.
 *
 *  If a certain backend is to be used, functions with the same interface as \p operator() but by the names \p CPU, \p CU,
 *  \p CL, \p OMP exists for CPU, CUDA, OpenCL and OpenMP respectively.
 *
 *  The MapArray skeleton also includes a pointer to an Environment which includes the devices available to execute on.
 */
template <typename MapArrayFunc>
class MapArray
{

public:

   MapArray(MapArrayFunc* mapArrayFunc);

   ~MapArray();

   /*!
    *  If the Mapping function supports a constant, set it with this function before a call.
    */
   template <typename U>
   void setConstant(U constant1)
   {
      m_mapArrayFunc->setConstant(constant1);
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
   MapArrayFunc* m_mapArrayFunc;

   /*! this is the pointer to array of execution plans in multiExecPlan scenario. Only one of them should be activated at a time considering current data locality */
   ExecPlan *m_execPlanMulti;

   /*! this is the pointer to execution plan that is active and should be used by implementations to check numOmpThreads and cudaBlocks etc. */
   ExecPlan *m_execPlan;

   /*! this is the default execution plan that gets created on initialization */
   ExecPlan m_defPlan;

public:
   template <typename T>
   void operator()(Vector<T>& input1, Vector<T>& input2, Vector<T>& output, int groupId = -1);

   template <typename T>
   void operator()(Vector<T>& input1, Matrix<T>& input2, Matrix<T>& output);

   template <typename T>
   void operator()(Vector<T>& input1, Matrix<T>& input2, Vector<T>& output);

   template <typename T>
   void operator()(Vector<T>& input1, SparseMatrix<T>& input2, Vector<T>& output);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void operator()(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin);

public:
   template <typename T>
   void CPU(Vector<T>& input1, Vector<T>& input2, Vector<T>& output);

   template <typename T>
   void CPU(Vector<T>& input1, Matrix<T>& input2, Matrix<T>& output);

   template <typename T>
   void CPU(Vector<T>& input1, Matrix<T>& input2, Vector<T>& output);

   template <typename T>
   void CPU(Vector<T>& input1, SparseMatrix<T>& input2, Vector<T>& output);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin);

#ifdef SKEPU_OPENMP
public:
   template <typename T>
   void OMP(Vector<T>& input1, Vector<T>& input2, Vector<T>& output);

   template <typename T>
   void OMP(Vector<T>& input1, Matrix<T>& input2, Matrix<T>& output);

   template <typename T>
   void OMP(Vector<T>& input1, Matrix<T>& input2, Vector<T>& output);

   template <typename T>
   void OMP(Vector<T>& input1, SparseMatrix<T>& input2, Vector<T>& output);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin);
#endif

#ifdef SKEPU_CUDA
public:
   template <typename T>
   void CU(Vector<T>& input1, Vector<T>& input2, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CU(Vector<T>& input1, Matrix<T>& input2, Matrix<T>& output, int useNumGPU = 1);

   template <typename T>
   void CU(Vector<T>& input1, Matrix<T>& input2, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CU(Vector<T>& input1, SparseMatrix<T>& input2, Vector<T>& output, int useNumGPU = 1);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU = 1);

private:
   unsigned int cudaDeviceID;

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void mapArraySingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, unsigned int deviceID);
#endif

#ifdef SKEPU_OPENCL
public:
   template <typename T>
   void CL(Vector<T>& input1, Vector<T>& input2, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CL(Vector<T>& input1, Matrix<T>& input2, Matrix<T>& output, int useNumGPU = 1);

   template <typename T>
   void CL(Vector<T>& input1, Matrix<T>& input2, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CL(Vector<T>& input1, SparseMatrix<T>& input2, Vector<T>& output, int useNumGPU = 1);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU = 1);

private:
   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void mapArrayNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, size_t numDevices);

private:
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_CL;

   void createOpenCLProgram();

#endif

};


}

#include "src/maparray.inl"

#include "src/maparray_cpu.inl"

#ifdef SKEPU_OPENMP
#include "src/maparray_omp.inl"
#endif

#ifdef SKEPU_OPENCL
#include "src/maparray_cl.inl"
#endif

#ifdef SKEPU_CUDA
#include "src/maparray_cu.inl"
#endif

#endif

