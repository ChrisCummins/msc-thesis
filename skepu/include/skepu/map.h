/*! \file map.h
 *  \brief Contains a class declaration for the Map skeleton.
 */

#ifndef MAP_H
#define MAP_H

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
 *  \class Map
 *
 *  \brief A class representing the Map skeleton.
 *
 *  This class defines the Map skeleton, a calculation pattern where a user function is applied to each element of an input
 *  range. Once the Map object is instantiated, it is meant to be used as a function and therefore overloading
 *  \p operator(). There are several overloaded versions of this operator that can be used depending on how many elements
 *  the mapping function uses (one, two or three). There are also variants which takes iterators as inputs and those that
 *  takes whole containers (vectors, matrices). The container variants are merely wrappers for the functions which takes iterators as parameters.
 *
 *  If a certain backend is to be used, functions with the same interface as \p operator() but by the names \p CPU, \p CU,
 *  \p CL, \p OMP exists for CPU, CUDA, OpenCL and OpenMP respectively.
 *
 *  The Map skeleton also includes a pointer to an Environment which includes the devices available to execute on.
 */
template <typename MapFunc>
class Map
{

public:

   Map(MapFunc* mapFunc);

   ~Map();

   /*!
    *  If the Mapping function supports a constant, set it with this function before a call.
    */
   template <typename U>
   void setConstant(U constant1)
   {
      m_mapFunc->setConstant(constant1);
   }

   /*!
    *  Makes sure all operations on devices are finished.
    */
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

   /*! this is the pointer to array of execution plans in multiExecPlan scenario. Only one of them should be activated at a time considering current data locality */
   ExecPlan *m_execPlanMulti;

   /*! this is the pointer to execution plan that is active and should be used by implementations to check numOmpThreads and cudaBlocks etc. */
   ExecPlan *m_execPlan;

   /*! this is the default execution plan that gets created on initialization */
   ExecPlan m_defPlan;

public:
   template <typename T>
   void operator()(Vector<T>& input);

   template <typename T>
   void operator()(Vector<T>& input, Vector<T>& output, int groupId = -1);

   template <typename T>
   void operator()(Matrix<T>& input);

   template <typename T>
   void operator()(Matrix<T>& input, Matrix<T>& output);

   template <typename InputIterator>
   void operator()(InputIterator inputBegin, InputIterator inputEnd);

   template <typename InputIterator, typename OutputIterator>
   void operator()(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin);

   template <typename T>
   void operator()(Vector<T>& input1, Vector<T>& input2, Vector<T>& output);

   template <typename T>
   void operator()(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& output);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void operator()(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin);

   template <typename T>
   void operator()(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, Vector<T>& output);

   template <typename T>
   void operator()(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, Matrix<T>& output);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
   void operator()(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin);

   template <typename T>
   void operator()(SparseMatrix<T>& input);

   template <typename T>
   void operator()(SparseMatrix<T>& input, SparseMatrix<T>& output);

   template <typename T>
   void operator()(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& output);

   template <typename T>
   void operator()(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& input3, SparseMatrix<T>& output);


public:
   template <typename T>
   void CPU(Vector<T>& input);

   template <typename T>
   void CPU(Vector<T>& input, Vector<T>& output);

   template <typename T>
   void CPU(Matrix<T>& input);

   template <typename T>
   void CPU(Matrix<T>& input, Matrix<T>& output);

   template <typename InputIterator>
   void CPU(InputIterator inputBegin, InputIterator inputEnd);

   template <typename InputIterator, typename OutputIterator>
   void CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin);

   template <typename T>
   void CPU(Vector<T>& input1, Vector<T>& input2, Vector<T>& output);

   template <typename T>
   void CPU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& output);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin);

   template <typename T>
   void CPU(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, Vector<T>& output);

   template <typename T>
   void CPU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, Matrix<T>& output);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
   void CPU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin);

   template <typename T>
   void CPU(SparseMatrix<T>& input);

   template <typename T>
   void CPU(SparseMatrix<T>& input, SparseMatrix<T>& output);

   template <typename T>
   void CPU(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& output);

   template <typename T>
   void CPU(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& input3, SparseMatrix<T>& output);

#ifdef SKEPU_OPENMP
public:
   template <typename T>
   void OMP(Vector<T>& input);

   template <typename T>
   void OMP(Vector<T>& input, Vector<T>& output);

   template <typename T>
   void OMP(Matrix<T>& input);

   template <typename T>
   void OMP(Matrix<T>& input, Matrix<T>& output);

   template <typename InputIterator>
   void OMP(InputIterator inputBegin, InputIterator inputEnd);

   template <typename InputIterator, typename OutputIterator>
   void OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin);

   template <typename T>
   void OMP(Vector<T>& input1, Vector<T>& input2, Vector<T>& output);

   template <typename T>
   void OMP(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& output);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin);

   template <typename T>
   void OMP(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, Vector<T>& output);

   template <typename T>
   void OMP(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, Matrix<T>& output);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
   void OMP(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin);

   template <typename T>
   void OMP(SparseMatrix<T>& input);

   template <typename T>
   void OMP(SparseMatrix<T>& input, SparseMatrix<T>& output);

   template <typename T>
   void OMP(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& output);

   template <typename T>
   void OMP(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& input3, SparseMatrix<T>& output);
#endif

#ifdef SKEPU_CUDA
public:
   template <typename T>
   void CU(Vector<T>& input, int useNumGPU = 1);

   template <typename T>
   void CU(Vector<T>& input, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CU(Matrix<T>& input, int useNumGPU = 1);

   template <typename T>
   void CU(Matrix<T>& input, Matrix<T>& output, int useNumGPU = 1);

   template <typename InputIterator>
   void CU(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU = 1);

   template <typename InputIterator, typename OutputIterator>
   void CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU = 1);

   template <typename InputIterator, typename OutputIterator>
   void CU_2(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU = 1);

   template <typename T>
   void CU(Vector<T>& input1, Vector<T>& input2, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& output, int useNumGPU = 1);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU = 1);

   template <typename T>
   void CU(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CU(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, Matrix<T>& output, int useNumGPU = 1);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
   void CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, int useNumGPU = 1);

   template <typename T>
   void CU(SparseMatrix<T>& input, int useNumGPU = 1);

   template <typename T>
   void CU(SparseMatrix<T>& input, SparseMatrix<T>& output, int useNumGPU = 1);

   template <typename T>
   void CU(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& output, int useNumGPU = 1);

   template <typename T>
   void CU(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& input3, SparseMatrix<T>& output, int useNumGPU = 1);

private:
   unsigned int cudaDeviceID;

   template <typename T>
   void mapSingleThread_CU(SparseMatrix<T> &input, SparseMatrix<T> &output, unsigned int deviceID);

   template <typename T>
   void mapSingleThread_CU(SparseMatrix<T> &input1, SparseMatrix<T> &input2, SparseMatrix<T> &output, unsigned int deviceID);

   template <typename T>
   void mapSingleThread_CU(SparseMatrix<T> &input1, SparseMatrix<T> &input2, SparseMatrix<T> &input3, SparseMatrix<T> &output, unsigned int deviceID);

   template <typename InputIterator, typename OutputIterator>
   void mapSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, unsigned int deviceID);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void mapSingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, unsigned int deviceID);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
   void mapSingleThread_CU(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, unsigned int deviceID);
#endif

#ifdef SKEPU_OPENCL
public:
   template <typename T>
   void CL(Vector<T>& input, int useNumGPU = 1);

   template <typename T>
   void CL(Vector<T>& input, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CL(Matrix<T>& input, int useNumGPU = 1);

   template <typename T>
   void CL(Matrix<T>& input, Matrix<T>& output, int useNumGPU = 1);

   template <typename InputIterator>
   void CL(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU = 1);

   template <typename InputIterator, typename OutputIterator>
   void CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, int useNumGPU = 1);

   template <typename T>
   void CL(Vector<T>& input1, Vector<T>& input2, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CL(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& output, int useNumGPU = 1);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, int useNumGPU = 1);

   template <typename T>
   void CL(Vector<T>& input1, Vector<T>& input2, Vector<T>& input3, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CL(Matrix<T>& input1, Matrix<T>& input2, Matrix<T>& input3, Matrix<T>& output, int useNumGPU = 1);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
   void CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, int useNumGPU = 1);

   template <typename T>
   void CL(SparseMatrix<T>& input, int useNumGPU = 1);

   template <typename T>
   void CL(SparseMatrix<T>& input, SparseMatrix<T>& output, int useNumGPU = 1);

   template <typename T>
   void CL(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& output, int useNumGPU = 1);

   template <typename T>
   void CL(SparseMatrix<T>& input1, SparseMatrix<T>& input2, SparseMatrix<T>& input3, SparseMatrix<T>& output, int useNumGPU = 1);

private:
   template <typename InputIterator, typename OutputIterator>
   void mapNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, size_t numDevices);

   template <typename Input1Iterator, typename Input2Iterator, typename OutputIterator>
   void mapNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, OutputIterator outputBegin, size_t numDevices);

   template <typename Input1Iterator, typename Input2Iterator, typename Input3Iterator, typename OutputIterator>
   void mapNumDevices_CL(Input1Iterator input1Begin, Input1Iterator input1End, Input2Iterator input2Begin, Input2Iterator input2End, Input3Iterator input3Begin, Input3Iterator input3End, OutputIterator outputBegin, size_t numDevices);

private:
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_CL;

   void createOpenCLProgram();
#endif

};

}

#include "src/map.inl"

#include "src/map_sparse.inl"

#include "src/map_cpu.inl"

#ifdef SKEPU_OPENMP
#include "src/map_omp.inl"
#endif

#ifdef SKEPU_OPENCL
#include "src/map_cl.inl"
#endif

#ifdef SKEPU_CUDA
#include "src/map_cu.inl"
#endif

#endif

