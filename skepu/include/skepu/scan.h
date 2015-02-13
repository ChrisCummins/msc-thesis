/*! \file scan.h
 *  \brief Contains a class declaration for the Scan skeleton.
 */

#ifndef SCAN_H
#define SCAN_H

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
 *  Enumeration of the two types of Scan that can be performed: Inclusive and Exclusive.
 */
enum ScanType
{
   INCLUSIVE,
   EXCLUSIVE
};

/*!
 *  \ingroup skeletons
 */

/*!
 *  \class Scan
 *
 *  \brief A class representing the Scan skeleton.
 *
 *  This class defines the Scan skeleton, also known as prefix sum. It is related to the Scan operation
 *  but instead of producing a single scalar result it produces an output vector of the same length as the
 *  input with its elements being the reduction of itself all elements preceding it in the input. For example the
 *  input vector [4 3 7 6 9] would produce the result vector [4 7 14 20 29]. The Scan operation can either include
 *  or exclude the current element. It can be either inclusive or exclusive. In the previous example a inclusive
 *  scan was performed, the exclusive result would be [0 4 7 14 20]. Exclusive scan is sometimes called prescan.
 *  This Scan skeleton supports both variants by adding a parameter to the function calls, default is inclusive.
 *
 *  Once instantiated, it is meant to be used as a function and therefore overloading
 *  \p operator(). There are a few overloaded variants of this operator depending on if a seperate output vector is provided
 *  or if vectors or iterators are used as parameters.
 *
 *  If a certain backend is to be used, functions with the same interface as \p operator() but by the names \p CPU, \p CU,
 *  \p CL, \p OMP exists for CPU, CUDA, OpenCL and OpenMP respectively.
 *
 *  The Scan skeleton also includes a pointer to an Environment which, in turn, includes the devices available to execute on.
 */
template <typename ScanFunc>
class Scan
{

public:

   Scan(ScanFunc* scanFunc);

   ~Scan();

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
   ScanFunc* m_scanFunc;

   /*! this is the pointer to array of execution plans in multiExecPlan scenario. Only one of them should be activated at a time considering current data locality */
   ExecPlan *m_execPlanMulti;

   /*! this is the pointer to execution plan that is active and should be used by implementations to check numOmpThreads and cudaBlocks etc. */
   ExecPlan *m_execPlan;

   /*! this is the default execution plan that gets created on initialization */
   ExecPlan m_defPlan;

public:
   template <typename T>
   void operator()(Vector<T>& input, ScanType type, T init = T());

   template <typename T>
   void operator()(Matrix<T>& input, ScanType type, T init = T());

   template <typename InputIterator>
   void operator()(InputIterator inputBegin, InputIterator inputEnd, ScanType type, typename InputIterator::value_type init = typename InputIterator::value_type());

   template <typename T>
   void operator()(Vector<T>& input, Vector<T>& output, ScanType type, T init = T());

   template <typename T>
   void operator()(Matrix<T>& input, Matrix<T>& output, ScanType type, T init = T());

   template <typename InputIterator, typename OutputIterator>
   void operator()(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init = typename InputIterator::value_type());

public:
   template <typename T>
   void CPU(Vector<T>& input, ScanType type = INCLUSIVE, T init = T());

   template <typename T>
   void CPU(Matrix<T>& input, ScanType type, T init = T());

   template <typename InputIterator>
   void CPU(InputIterator inputBegin, InputIterator inputEnd, ScanType type = INCLUSIVE, typename InputIterator::value_type init = typename InputIterator::value_type());

   template <typename T>
   void CPU(Vector<T>& input, Vector<T>& output, ScanType type = INCLUSIVE, T init = T());

   template <typename T>
   void CPU(Matrix<T>& input, Matrix<T>& output, ScanType type, T init = T());

   template <typename InputIterator, typename OutputIterator>
   void CPU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type = INCLUSIVE, typename InputIterator::value_type init = typename InputIterator::value_type());

#ifdef SKEPU_OPENMP
public:
   template <typename T>
   void OMP(Vector<T>& input, ScanType type = INCLUSIVE, T init = T());

   template <typename T>
   void OMP(Matrix<T>& input, ScanType type, T init = T());

   template <typename InputIterator>
   void OMP(InputIterator inputBegin, InputIterator inputEnd, ScanType type = INCLUSIVE, typename InputIterator::value_type init = typename InputIterator::value_type());

   template <typename T>
   void OMP(Vector<T>& input, Vector<T>& output, ScanType type = INCLUSIVE, T init = T());

   template <typename T>
   void OMP(Matrix<T>& input, Matrix<T>& output, ScanType type, T init = T());

   template <typename InputIterator, typename OutputIterator>
   void OMP(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type = INCLUSIVE, typename InputIterator::value_type init = typename InputIterator::value_type());
#endif

#ifdef SKEPU_CUDA
public:
   template <typename T>
   void CU(Vector<T>& input, ScanType type = INCLUSIVE, T init = T(), int useNumGPU = 1);

//    template <typename T>
//    void CU(Matrix<T>& input, ScanType type, T init = T(), int useNumGPU = 1);

   template <typename InputIterator>
   void CU(InputIterator inputBegin, InputIterator inputEnd, ScanType type = INCLUSIVE, typename InputIterator::value_type init = typename InputIterator::value_type(), int useNumGPU = 1);

   template <typename T>
   void CU(Vector<T>& input, Vector<T>& output, ScanType type = INCLUSIVE, T init = T(), int useNumGPU = 1);

//    template <typename T>
//    void CU(Matrix<T>& input, Matrix<T>& output, ScanType type, T init = T(), int useNumGPU = 1);

   template <typename InputIterator, typename OutputIterator>
   void CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type = INCLUSIVE, typename InputIterator::value_type init = typename InputIterator::value_type(), int useNumGPU = 1);

private:
   unsigned int cudaDeviceID;
   
   template <typename InputIterator, typename OutputIterator>
   void scanSingleThread_CU(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, unsigned int deviceID);

   template <typename T>
   T scanLargeVectorRecursively_CU(DeviceMemPointer_CU<T>* input, DeviceMemPointer_CU<T>* output, std::vector<DeviceMemPointer_CU<T>*>& blockSums, size_t numElements, unsigned int level, ScanType type, T init, unsigned int deviceID);

#endif

#ifdef SKEPU_OPENCL
public:
   template <typename T>
   void CL(Vector<T>& input, ScanType type = INCLUSIVE, T init = T(), int useNumGPU = 1);

//    template <typename T>
//    void CL(Matrix<T>& input, ScanType type, T init = T(), int useNumGPU = 1);

   template <typename InputIterator>
   void CL(InputIterator inputBegin, InputIterator inputEnd, ScanType type = INCLUSIVE, typename InputIterator::value_type init = typename InputIterator::value_type(), int useNumGPU = 1);

   template <typename T>
   void CL(Vector<T>& input, Vector<T>& output, ScanType type = INCLUSIVE, T init = T(), int useNumGPU = 1);

//    template <typename T>
//    void CL(Matrix<T>& input, Matrix<T>& output, ScanType type, T init = T(), int useNumGPU = 1);

   template <typename InputIterator, typename OutputIterator>
   void CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type = INCLUSIVE, typename InputIterator::value_type init = typename InputIterator::value_type(), int useNumGPU = 1);

private:
   template <typename InputIterator, typename OutputIterator>
   void scanSingle_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, unsigned int deviceID);

   template <typename InputIterator, typename OutputIterator>
   void scanNumDevices_CL(InputIterator inputBegin, InputIterator inputEnd, OutputIterator outputBegin, ScanType type, typename InputIterator::value_type init, size_t numDevices);

   template <typename T>
   T scanLargeVectorRecursively_CL(DeviceMemPointer_CL<T>* input, DeviceMemPointer_CL<T>* output, std::vector<DeviceMemPointer_CL<T>*>& blockSums, size_t numElements, unsigned int level, ScanType type, T init, unsigned int deviceID);

private:
   std::vector<std::pair<cl_kernel, Device_CL*> > m_scanKernels_CL;
   std::vector<std::pair<cl_kernel, Device_CL*> > m_scanUpdateKernels_CL;
   std::vector<std::pair<cl_kernel, Device_CL*> > m_scanAddKernels_CL;

   void createOpenCLProgram();
#endif

};

}

#include "src/scan.inl"

#include "src/scan_cpu.inl"

#ifdef SKEPU_OPENMP
#include "src/scan_omp.inl"
#endif

#ifdef SKEPU_OPENCL
#include "src/scan_cl.inl"
#endif

#ifdef SKEPU_CUDA
#include "src/scan_cu.inl"
#endif

#endif

