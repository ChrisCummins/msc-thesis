/*! \file generate.h
 *  \brief Contains a class declaration for the Generate skeleton.
 */

#ifndef GENERATE_H
#define GENERATE_H

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
 *  \ingroup skeletons
 */

/*!
 *  \class Generate
 *
 *  \brief A class representing the Generate skeleton.
 *
 *  This class defines the Generate skeleton, It can be used to generate a specified number
 *  of elements using a user defined function for both vector and matrix containers.
 *  It is similar to the Unary Map skeleton, applying
 *  a function to each element of a vector. But the Generate skeleton generates all the elements instead
 *  of copying the existing elements and applying a function on them.
 *
 *  Once the Generate object is instantiated, it is meant to be used as a function and therefore overloading
 *  \p operator(). There are different overloaded versions of this operator, one which takes iterators as inputs and one that
 *  takes whole vectors/matrix.
 *
 *  If a certain back end is to be used, functions with the same interface as \p operator() but by the names \p CPU, \p CU,
 *  \p CL, \p OMP exists for CPU, CUDA, OpenCL and OpenMP respectively.
 */
template <typename GenerateFunc>
class Generate
{

public:

   Generate(GenerateFunc* generateFunc);

   ~Generate();

   /*!
    *  This function sets the constant that can be used in the generate user function.
    */
   template <typename T>
   void setConstant(T constant1)
   {
      m_generateFunc->setConstant(constant1);
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
   GenerateFunc* m_generateFunc;

   /*! this is the pointer to array of execution plans in multiExecPlan scenario. Only one of them should be activated at a time considering current data locality */
   ExecPlan *m_execPlanMulti;

   /*! this is the pointer to execution plan that is active and should be used by implementations to check numOmpThreads and cudaBlocks etc. */
   ExecPlan *m_execPlan;

   /*! this is the default execution plan that gets created on initialization */
   ExecPlan m_defPlan;


public:
   template <typename T>
   void operator()(size_t numElements, Vector<T>& output);

   template <typename T>
   void operator()(size_t numRows, size_t numCols, Matrix<T>& output);

   template <typename OutputIterator>
   void operator()(size_t numElements, OutputIterator outputBegin);

public:
   template <typename T>
   void CPU(size_t numElements, Vector<T>& output);

   template <typename T>
   void CPU(size_t numRows, size_t numCols, Matrix<T>& output);

   template <typename OutputIterator>
   void CPU(size_t numElements, OutputIterator outputBegin);


#ifdef SKEPU_OPENMP
public:
   template <typename T>
   void OMP(size_t numElements, Vector<T>& output);

   template <typename T>
   void OMP(size_t numRows, size_t numCols, Matrix<T>& output);

   template <typename OutputIterator>
   void OMP(size_t numElements, OutputIterator outputBegin);

#endif

#ifdef SKEPU_CUDA
public:
   template <typename T>
   void CU(size_t numElements, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CU(size_t numRows, size_t numCols, Matrix<T>& output, int useNumGPU = 1);

   template <typename OutputIterator>
   void CU(size_t numElements, OutputIterator outputBegin, int useNumGPU = 1);

private:
   unsigned int cudaDeviceID;
   
   template <typename OutputIterator>
   void generateSingleThread_CU(size_t numElements, OutputIterator outputBegin, unsigned int deviceID);

#endif

#ifdef SKEPU_OPENCL
public:
   template <typename T>
   void CL(size_t numElements, Vector<T>& output, int useNumGPU = 1);

   template <typename T>
   void CL(size_t numRows, size_t numCols, Matrix<T>& output, int useNumGPU = 1);

   template <typename OutputIterator>
   void CL(size_t numElements, OutputIterator outputBegin, int useNumGPU = 1);

private:
   template <typename OutputIterator>
   void generateNumDevices_CL(size_t numElements, OutputIterator outputBegin, size_t numDevices);

private:
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_CL;

   void createOpenCLProgram();
#endif

};

}

#include "src/generate.inl"

#include "src/generate_cpu.inl"

#ifdef SKEPU_OPENMP
#include "src/generate_omp.inl"
#endif

#ifdef SKEPU_OPENCL
#include "src/generate_cl.inl"
#endif

#ifdef SKEPU_CUDA
#include "src/generate_cu.inl"
#endif

#endif

