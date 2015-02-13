/*! \file reduce.h
 *  \brief Contains a class declaration for the Reduce skeleton.
 */

#ifndef REDUCE_H
#define REDUCE_H

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
 *  \brief Can be used to specify the direction of reduce for 2D containers
 *
 *  Used in reduction operations for 2D containers.
 */
enum ReducePolicy
{
   REDUCE_ROW_WISE_ONLY,
   REDUCE_COL_WISE_ONLY
};

/*!
 *  \ingroup skeletons
 */

/*!
 *  \class Reduce
 *
 *  \brief A class representing the Reduce skeleton both for 1D and 2D reduce operation for 1D Vector, 2D Dense Matrix/Sparse matrices.
 *
 *  This class defines the Reduce skeleton which support following reduction operations:
 *  (a) (1D Reduction) Each element in the input range, yielding a scalar result by applying a commutative associative binary operator.
 *     Here we consider dense/sparse matrix as vector thus reducing all (non-zero) elements of the matrix.
 *  (b) (1D Reduction) Dense/Sparse matrix types: Where we reduce either row-wise or column-wise by applying a commutative associative binary operator. It returns
 *     a \em SkePU vector of results that corresponds to reduction on either dimension.
 *  (c) (2D Reduction) Dense/Sparse matrix types: Where we reduce both row- and column-wise by applying two commutative associative
 *     binary operators, one for row-wise reduction and one for column-wise reduction. It returns a scalar result.
 *  Two versions of this class are created using C++ partial class-template specialization to support
 *  (a) 1D reduction (where a "single" reduction operator is applied on all elements or to 1 direction for 2D Dense/Sparse matrix).
 *  (b) 2D reduction that works only for matrix (where two different reduction operations are used to reduce row-wise and column-wise separately.)
 *  Once instantiated, it is meant to be used as a function and therefore overloading
 *  \p operator(). The Reduce skeleton needs to be created with
 *  a 1 or 2 binary user function for 1D reduction and 2D reduction respectively.
 *
 *  If a certain backend is to be used, functions with the same interface as \p operator() but by the names \p CPU, \p CU,
 *  \p CL, \p OMP exists for CPU, CUDA, OpenCL and OpenMP respectively.
 *
 *  The Reduce skeleton also includes a pointer to an Environment which includes the devices available to execute on.
 */
template <typename ReduceFuncRowWise, typename ReduceFuncColWise = ReduceFuncRowWise>
class Reduce
{

public:

   Reduce(ReduceFuncRowWise* reduceFuncRowWise, ReduceFuncColWise* reduceFuncColWise);

   ~Reduce();

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
   ReduceFuncRowWise* m_reduceFuncRowWise;
   ReduceFuncColWise* m_reduceFuncColWise;

   /*! this is the pointer to array of execution plans in multiExecPlan scenario. Only one of them should be activated at a time considering current data locality */
   ExecPlan *m_execPlanMulti;

   /*! this is the pointer to execution plan that is active and should be used by implementations to check numOmpThreads and cudaBlocks etc. */
   ExecPlan *m_execPlan;

   /*! this is the default execution plan that gets created on initialization */
   ExecPlan m_defPlan;

public:
   template <typename T>
   T operator()(Matrix<T>& input);

   template <typename T>
   T operator()(SparseMatrix<T>& input);

public:
   template <typename T>
   T CPU(Matrix<T>& input);

   template <typename T>
   T CPU(SparseMatrix<T>& input);

#ifdef SKEPU_OPENMP
public:
   template <typename T>
   T OMP(Matrix<T>& input);

   template <typename T>
   T OMP(SparseMatrix<T>& input);

private:
   template <typename T>
   T ompVectorReduce(std::vector<T> &input, const size_t &numThreads);
#endif

#ifdef SKEPU_CUDA
public:
   template <typename T>
   T CU(Matrix<T>& input, int useNumGPU = 1);

   template <typename T>
   T CU(SparseMatrix<T>& input, int useNumGPU = 1);

private:
   unsigned int cudaDeviceID;

   template <typename T>
   T reduceSingleThread_CU(Matrix<T>& input, unsigned int deviceID);

   template <typename T>
   T reduceMultipleThreads_CU(Matrix<T>& input, size_t numDevices);

   template <typename T>
   T reduceSingleThread_CU(SparseMatrix<T>& input, unsigned int deviceID);

   template <typename T>
   T reduceMultipleThreads_CU(SparseMatrix<T>& input, size_t numDevices);

#endif

#ifdef SKEPU_OPENCL
public:
   template <typename T>
   T CL(Matrix<T>& input, int useNumGPU = 1);

   template <typename T>
   T CL(SparseMatrix<T>& input, int useNumGPU = 1);

private:
   template <typename T>
   T reduceSingle_CL(Matrix<T> &input, unsigned int deviceID);

   template <typename T>
   T reduceNumDevices_CL(Matrix<T> &input, size_t numDevices);

   template <typename T>
   T reduceSingle_CL(SparseMatrix<T> &input, unsigned int deviceID);

   template <typename T>
   T reduceNumDevices_CL(SparseMatrix<T> &input, size_t numDevices);

private:
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_CL_RowWise;
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_CL_ColWise;

   void createOpenCLProgram();
#endif

};





/*!
 *
 * \brief A specilalization of above class, used for 1D Reduce operation.
 * Please note that the class name is same. The only difference is
 * how you instantiate it either by passing 1 user function (i.e. 1D reduction)
 * or 2 user function (i.e. 2D reduction). See code examples for more information.
 */
template <typename ReduceFunc>
class Reduce<ReduceFunc, ReduceFunc>
{
public:
   Reduce(ReduceFunc* reduceFunc);

   ~Reduce();

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

   template <typename T>
   T operator()(SparseMatrix<T>& input);

   template <typename T>
   Vector<T> operator()(Matrix<T>& input, ReducePolicy reducePolicy);

   template <typename T>
   Vector<T> operator()(SparseMatrix<T>& input, ReducePolicy reducePolicy);

   template <typename InputIterator>
   typename InputIterator::value_type operator()(InputIterator inputBegin, InputIterator inputEnd);

public:
   template <typename T>
   T CPU(Vector<T>& input);

   template <typename T>
   T CPU(Matrix<T>& input);

   template <typename T>
   T CPU(SparseMatrix<T>& input);

   template <typename T>
   Vector<T> CPU(Matrix<T>& input, ReducePolicy reducePolicy);

   template <typename T>
   Vector<T> CPU(SparseMatrix<T>& input, ReducePolicy reducePolicy);

   template <typename InputIterator>
   typename InputIterator::value_type CPU(InputIterator inputBegin, InputIterator inputEnd);

#ifdef SKEPU_OPENMP
public:
   template <typename T>
   T OMP(Vector<T>& input);

   template <typename T>
   T OMP(Matrix<T>& input);

   template <typename T>
   T OMP(SparseMatrix<T>& input);

   template <typename T>
   Vector<T> OMP(Matrix<T>& input, ReducePolicy reducePolicy);

   template <typename T>
   Vector<T> OMP(SparseMatrix<T>& input, ReducePolicy reducePolicy);

   template <typename InputIterator>
   typename InputIterator::value_type OMP(InputIterator inputBegin, InputIterator inputEnd);
#endif

#ifdef SKEPU_CUDA
public:
   template <typename T>
   T CU(Vector<T>& input, int useNumGPU = 1);

   template <typename T>
   T CU(Matrix<T>& input, int useNumGPU = 1);

   template <typename T>
   T CU(SparseMatrix<T>& input, int useNumGPU = 1);

   template <typename T>
   Vector<T> CU(Matrix<T>& input, ReducePolicy reducePolicy, int useNumGPU = 1);

   template <typename T>
   Vector<T> CU(SparseMatrix<T>& input, ReducePolicy reducePolicy, int useNumGPU = 1);

   template <typename InputIterator>
   typename InputIterator::value_type CU(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU = 1);

private:
   int cudaDeviceID;

   template <typename InputIterator>
   typename InputIterator::value_type reduceSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID);

   template <typename T>
   T reduceSingleThread_CU(SparseMatrix<T>& input, unsigned int deviceID);

   template <typename T>
   void reduceSingleThreadOneDim_CU(Matrix<T>& input, unsigned int deviceID, Vector<T> &result);

   template <typename T>
   void reduceSingleThreadOneDim_CU(SparseMatrix<T>& input, unsigned int deviceID, Vector<T> &result);
#endif

#ifdef SKEPU_OPENCL
public:
   template <typename T>
   T CL(Vector<T>& input, int useNumGPU = 1);

   template <typename T>
   T CL(Matrix<T>& input, int useNumGPU = 1);

   template <typename T>
   T CL(SparseMatrix<T>& input, int useNumGPU = 1);

   template <typename T>
   Vector<T> CL(Matrix<T>& input, ReducePolicy reducePolicy, int useNumGPU = 1);

   template <typename T>
   Vector<T> CL(SparseMatrix<T>& input, ReducePolicy reducePolicy, int useNumGPU = 1);

   template <typename InputIterator>
   typename InputIterator::value_type CL(InputIterator inputBegin, InputIterator inputEnd, int useNumGPU = 1);

private:
   template <typename InputIterator>
   typename InputIterator::value_type reduceSingle_CL(InputIterator inputBegin, InputIterator inputEnd, unsigned int deviceID);

   template <typename T>
   T reduceSingle_CL(SparseMatrix<T>& input, unsigned int deviceID);

   template <typename T>
   void reduceSingleThreadOneDim_CL(Matrix<T>& input, unsigned int deviceID, Vector<T> &result);

   template <typename T>
   void reduceSingleThreadOneDim_CL(SparseMatrix<T>& input, unsigned int deviceID, Vector<T> &result);

private:
   std::vector<std::pair<cl_kernel, Device_CL*> > m_kernels_CL;

   void createOpenCLProgram();
#endif

};

}


#include "src/reduce_common.h"

#include "src/reduce.inl"
#include "src/reduce_2d.inl"

#include "src/reduce_cpu.inl"
#include "src/reduce_cpu_2d.inl"

#ifdef SKEPU_OPENMP
#include "src/reduce_omp.inl"
#include "src/reduce_omp_2d.inl"
#endif

#ifdef SKEPU_OPENCL
#include "src/reduce_cl.inl"
#include "src/reduce_cl_2d.inl"
#endif

#ifdef SKEPU_CUDA
#include "src/reduce_cu.inl"
#include "src/reduce_cu_2d.inl"
#endif





#endif

