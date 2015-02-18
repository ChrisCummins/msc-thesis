/// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

/// following define to enable/disable OpenMP implmentation to be used
#define SKEPU_OPENMP

/// following define to specify how many GPU devices can SkePU use... By default its 4...
#define MAX_GPU_DEVICES 1

/// following define to specify how many OpenMP threads SkePU should use for its OpenMP implementations... By default its max available...
#define SKEPU_OPENMP_THREADS 4

/// following define to specify how many OpenMP threads SkePU should use for its OpenMP implementations... By default its max available...
#define SKEPU_OPENMP_THREADS 4

/*! 
 * Enable following to specify how many GPU threads per block should be used with CUDA.
 * by default, SKEPU used max threads possible which could be a problem if a kernel
 * consumes lot of registers and thus cannot execute with that # of threads.
 */
// #define SKEPU_MAX_GPU_THREADS 512

/*!
 * following are some defines related to tuning framework...
 * REDO_MEASUREMENTS specifies that tuning should be (re-)done each time rather than loading from earlier executions...
 * DISCARD_FIRST_EXEC specifies that first execution hsould be discarded from training measurements... 
 * TRAINING_RUNS specifies how many training runs are made for each training point... 
 */
#define REDO_MEASUREMENTS
#define DISCARD_FIRST_EXEC
#define TRAINING_RUNS 3


#include <iostream>

#include "skepu/vector.h"
#include "skepu/mapreduce.h"
#include "skepu/tuner.h"

BINARY_FUNC(square_f, float, a, b,
            return (a*b);
           )

BINARY_FUNC(plus_f, float, a, b,
            return a+b;
           )




const int N = 10;
const int LOW_N = 5;
const int HIGH_N = 49999;


int main()
{
   skepu::MapReduce<square_f, plus_f> dotProduct(new square_f, new plus_f);

   size_t lowerBounds[] = {10, 10};
   size_t upperBounds[] = {50000, 50000};

   /*!
    * The tuner can work in two modes:
    * 1st mode: you can specify memory location hints for operand data for each skeleton call using memFlags
    */
//    int memFlags[] = {0, 0}; /*! 0 means that that operand resides on main memory, 1 for gpu memory... */
//    skepu::ExecPlan execPlan1 = skepu::Tuner<square_f, MAPREDUCE, plus_f>("dotproduct", 2, lowerBounds, upperBounds, memFlags)();
//    dotProduct.setExecPlan(execPlan1);
   
   skepu::ExecPlan execPlan[MAX_EXEC_PLANS];
   skepu::Tuner<square_f, MAPREDUCE, plus_f> tuner("dotproduct", 2, lowerBounds, upperBounds);
   tuner(execPlan);
   dotProduct.setExecPlan(execPlan);

//    std::cout << "Tuning statistics...\n";
//    std::cout << "maxDepth: "            << tuner.stats.maxDepth << "\n";
//    std::cout << "numLeafClosedRanges: " << tuner.stats.numLeafClosedRanges << "\n";
//    std::cout << "numLeafTotalRanges: "  << tuner.stats.numLeafTotalRanges << "\n";
//    std::cout << "numTotalRanges: "      << tuner.stats.numTotalRanges << "\n";
//    std::cout << "numTrainingPoints: "   << tuner.stats.numTrainingPoints << "\n";
//    std::cout << "numTrainingExec: "     << tuner.stats.numTrainingExec << "\n";

   /// generate random testing points...
   double SIZES[N];
   double GAP = (log(HIGH_N)-log(LOW_N))/N;
   SIZES[0] = LOW_N*exp(GAP);
   for(int i=1; i<N; ++i)
   {
      SIZES[i] = (SIZES[i-1]*exp(GAP));
   }
   //rounding
   for(int i=0; i<N; ++i)
   {
      SIZES[i] = round(SIZES[i]);
   }

   for(int r=0; r<N; ++r)
   {
      int size = SIZES[r];

      skepu::Vector<float> v0(size);
      skepu::Vector<float> v1(size);
      
      v0.randomize(0, 5);
      v1.randomize(0, 3);

      double result;

      /// execution on a single CPU
//       result = dotProduct.CPU(v0, v1);
      
      /// execution on OpenMP
//       result = dotProduct.OMP(v0, v1);
      
      /// execution on CPU
//       result = dotProduct.CU(v0, v1);
      
      /// execution guided by the execution plan(s)
      result = dotProduct(v0, v1);
      
      std::cerr << "Size: " << size << ", Result: " << result << "\n";
   }
   return 0;
}