#ifndef SKEPU_TUNER
#define SKEPU_TUNER

#include <cstdlib>
#include <iostream>
#include <cassert>






#include "skepu/map.h"
#include "skepu/reduce.h"
#include "skepu/mapreduce.h"
#include "skepu/mapoverlap.h"
#include "skepu/maparray.h"
#include "skepu/scan.h"

#include <vector>

#include "skepu/src/trainer.h"
#include "skepu/src/timer.h"

enum SkeletonType
{
   MAP,
   REDUCE,
   MAPREDUCE,
   SCAN,
   MAPARRAY,
   MAPOVERLAP
};


namespace skepu
{

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for Map skeleton and sequential CPU implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void cpu_tune_wrapper_map(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens >= 1 && actDimens <= 4);

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = td->problemSize[0];
      vecArr[i].resize(sizes[i]);
   }

   double commCost[MAX_EXEC_PLANS];
#ifdef SKEPU_CUDA
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   commCost[0] = 0.0;
   double commCostPerOp = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
   // the user can specify flag hints for operands memory location
   bool singlePlan = (td->extra->memUp != NULL && td->extra->memDown != NULL);
   if(singlePlan)
   {
      assert(nImpls == 1);
      
      int *memUpFlags = td->extra->memUp;
      int *memDownFlags = td->extra->memDown;
      for(unsigned int i=0; i<actDimens; ++i)
      {
         if(i < (actDimens - 1) && memUpFlags[i] == 1) 
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
         else if(i == (actDimens - 1) && memDownFlags[0] == 1) 
            commCost[0] += bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
   }
   else
   {
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = commCostPerOp;
      }
      if(nImpls > 3)
      {
         commCost[3] = 0.0;
         commCost[4] = commCostPerOp;
         commCost[5] = commCostPerOp * 2;
      }
      if(nImpls > 6)
      {
         commCost[6] = 0.0;
         commCost[7] = commCostPerOp;
         commCost[8] = commCostPerOp * 2;
         commCost[9] = commCostPerOp * 3;
      }
   }
#else
   /*! if CUDA is not enabled we cannot consider the other cases */
   commCost[0] = 0.0;
   assert(nImpls == 1);
#endif

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::Map<StructType> mapTest(userFunc);

   timer.start();

   if(actDimens == 1)
      mapTest.CPU(vecArr[0]);
   else if(actDimens == 2)
      mapTest.CPU(vecArr[0],vecArr[1]);
   else if(actDimens == 3)
      mapTest.CPU(vecArr[0],vecArr[1], vecArr[2]);
   else if(actDimens == 4)
      mapTest.CPU(vecArr[0],vecArr[1], vecArr[2], vecArr[3]);
   else
      assert(false);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*CPU* map size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for Reduce skeleton and sequential CPU implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void cpu_tune_wrapper_reduce(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens == 1);

   // to ensure that compiler does not optimize these calls away as retVal is not used anywhere...
   volatile typename StructType::TYPE retVal;

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = ((actDimens != dimens)? td->problemSize[0] : td->problemSize[i]);
      vecArr[i].resize(sizes[i]);
   }

   double commCost[MAX_EXEC_PLANS];
#ifdef SKEPU_CUDA
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   commCost[0] = 0.0;
   // the user can specify flag hints for operands memory location
   bool singlePlan = (td->extra->memUp != NULL);
   if(singlePlan)
   {
      assert(nImpls == 1);
      
      int *memUpFlags = td->extra->memUp;
      for(unsigned int i=0; i<actDimens; ++i)
      {
         if(memUpFlags[i] == 1) 
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
   }
   else
   {
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
      }
   }
#else
   /*! if CUDA is not enabled we cannot consider the other cases */
   commCost[0] = 0.0;
   assert(nImpls == 1);
#endif

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::Reduce<StructType> redTest(userFunc);

   timer.start();

   retVal = redTest.CPU(vecArr[0]);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*CPU* reduce size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for MapOverlap skeleton and sequential CPU implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void cpu_tune_wrapper_mapoverlap(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens >=1 && actDimens <= 2);

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = ((actDimens != dimens)? td->problemSize[0] : td->problemSize[i]);
      vecArr[i].resize(sizes[i]);
   }

   double commCost[MAX_EXEC_PLANS];
#ifdef SKEPU_CUDA
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   commCost[0] = 0.0;
   // the user can specify flag hints for operands memory location
   bool singlePlan = (td->extra->memUp != NULL && td->extra->memDown != NULL);
   if(singlePlan)
   {
      assert(nImpls == 1);
      
      int *memUpFlags = td->extra->memUp;
      int *memDownFlags = td->extra->memDown;
      for(unsigned int i=0; i<actDimens; ++i)
      {
         if(i < (actDimens - 1) && memUpFlags[i] == 1) 
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
         else if(i == (actDimens - 1) && memDownFlags[0] == 1) 
            commCost[0] += bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
   }
   else
   {   
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
      }
   }
#else
   /*! if CUDA is not enabled we cannot consider the other cases */
   commCost[0] = 0.0;
   assert(nImpls == 1);
#endif

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::MapOverlap<StructType> mapOverTest(userFunc);

   timer.start();

   if(actDimens == 1)
      mapOverTest.CPU(vecArr[0]);
   else if(actDimens == 2)
      mapOverTest.CPU(vecArr[0], vecArr[1]);
   else
      assert(false);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*CPU* mapoverlap size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for MapArray skeleton and sequential CPU implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void cpu_tune_wrapper_maparray(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens >= 1 && dimens <= 2 && actDimens == 3);

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];

   sizes[0] = td->problemSize[0];
   sizes[1] = (dimens == 1)? td->problemSize[0] : td->problemSize[1];
   sizes[2] = (dimens == 1)? td->problemSize[0] : td->problemSize[1];

   for(unsigned int i=0; i<actDimens; ++i)
   {
      vecArr[i].resize(sizes[i]);
   }

   double commCost[MAX_EXEC_PLANS];
#ifdef SKEPU_CUDA
   assert(sizes[0] == sizes[1] && sizes[1] == sizes[2]);

   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   commCost[0] = 0.0;
   // the user can specify flag hints for operands memory location
   bool singlePlan = (td->extra->memUp != NULL && td->extra->memDown != NULL);
   if(singlePlan)
   {
      assert(nImpls == 1);
      
      int *memUpFlags = td->extra->memUp;
      int *memDownFlags = td->extra->memDown;
      for(unsigned int i=0; i<actDimens; ++i)
      {
         if(i < (actDimens - 1) && memUpFlags[i] == 1) 
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
         else if(i == (actDimens - 1) && memDownFlags[0] == 1) 
            commCost[0] += bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
   }
   else
   {
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
      if(nImpls > 3)
      {
         assert(sizes[0] == sizes[1]); /*! TODO: fix it in future */
         commCost[3] = 0.0;
         commCost[4] = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[1]));
         commCost[5] = commCost[2] + bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[1]));
      }
   }
#else
   /*! if CUDA is not enabled we cannot consider the other cases */
   commCost[0] = 0.0;
   assert(nImpls == 1);
#endif

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::MapArray<StructType> mapArrTest(userFunc);

   timer.start();

   mapArrTest.CPU(vecArr[0], vecArr[1], vecArr[2]);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*CPU* maparray size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}


/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for MapReduce skeleton and sequential CPU implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void cpu_tune_wrapper_mapreduce(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens >= 1 && actDimens <= 3);

   // to ensure that compiler does not optimize these calls away as retVal is not used anywhere...
   volatile typename StructType::TYPE retVal;

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = td->problemSize[0];
      vecArr[i].resize(sizes[i]);
   }

   double commCost[MAX_EXEC_PLANS];
#ifdef SKEPU_CUDA
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   commCost[0] = 0.0;
   double commCostPerOp = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
   // the user can specify flag hints for operands memory location
   bool singlePlan = (td->extra->memUp != NULL);
   if(singlePlan)
   {
      assert(nImpls == 1);
      
      int *memUpFlags = td->extra->memUp;
      for(unsigned int i=0; i<actDimens; ++i)
      {
         if(memUpFlags[i] == 1) 
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
   }
   else
   {
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = commCostPerOp;
      }
      if(nImpls > 3)
      {
         commCost[3] = 0.0;
         commCost[4] = commCostPerOp;
         commCost[5] = commCostPerOp * 2;
      }
      if(nImpls > 6)
      {
         commCost[6] = 0.0;
         commCost[7] = commCostPerOp;
         commCost[8] = commCostPerOp * 2;
         commCost[9] = commCostPerOp * 3;
      }
   }
#else
   /*! if CUDA is not enabled we cannot consider the other cases */
   commCost[0] = 0.0;
   assert(nImpls == 1);
#endif

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   StructType2 *userFunc2 = new StructType2;
   if(td->callBackFunctionMapReduce != NULL)
      td->callBackFunctionMapReduce(userFunc, userFunc2, sizes, actDimens);

   skepu::MapReduce<StructType, StructType2> mapRedTest(userFunc, userFunc2);

   timer.start();

   if(actDimens == 1)
      retVal = mapRedTest.CPU(vecArr[0]);
   else if(actDimens == 2)
      retVal = mapRedTest.CPU(vecArr[0],vecArr[1]);
   else if(actDimens == 3)
      retVal = mapRedTest.CPU(vecArr[0],vecArr[1], vecArr[2]);
   else
      assert(false);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*CPU* mapreduce size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}


/// the following section contains function that can train OpenMP implementations. Only enabled when OpenMP is enabled in SkePU library


#ifdef SKEPU_OPENMP

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for Map skeleton and parallel OpenMP implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void omp_tune_wrapper_map(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens >= 1 && actDimens <= 4);

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = td->problemSize[0];
      vecArr[i].resize(sizes[i]);
   }

   double commCost[MAX_EXEC_PLANS];
#ifdef SKEPU_CUDA
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   commCost[0] = 0.0;
   double commCostPerOp = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
   
   // the user can specify flag hints for operands memory location
   bool singlePlan = (td->extra->memUp != NULL && td->extra->memDown != NULL);
   if(singlePlan)
   {
      assert(nImpls == 1);
      
      int *memUpFlags = td->extra->memUp;
      int *memDownFlags = td->extra->memDown;
      for(unsigned int i=0; i<actDimens; ++i)
      {
         if(i < (actDimens - 1) && memUpFlags[i] == 1) 
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
         else if(i == (actDimens - 1) && memDownFlags[0] == 1) 
            commCost[0] += bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
   }
   else
   {
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = commCostPerOp;
      }
      if(nImpls > 3)
      {
         commCost[3] = 0.0;
         commCost[4] = commCostPerOp;
         commCost[5] = commCostPerOp * 2;
      }
      if(nImpls > 6)
      {
         commCost[6] = 0.0;
         commCost[7] = commCostPerOp;
         commCost[8] = commCostPerOp * 2;
         commCost[9] = commCostPerOp * 3;
      }
   }
#else
   /*! if CUDA is not enabled we cannot consider the other cases */
   commCost[0] = 0.0;
   assert(nImpls == 1);
#endif

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::Map<StructType> mapTest(userFunc);

   timer.start();

   if(actDimens == 1)
      mapTest.OMP(vecArr[0]);
   else if(actDimens == 2)
      mapTest.OMP(vecArr[0],vecArr[1]);
   else if(actDimens == 3)
      mapTest.OMP(vecArr[0],vecArr[1], vecArr[2]);
   else if(actDimens == 4)
      mapTest.OMP(vecArr[0],vecArr[1], vecArr[2], vecArr[3]);
   else
      assert(false);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*OpenMP* map size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for Reduce skeleton and parallel OpenMP implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void omp_tune_wrapper_reduce(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens == 1);

   // to ensure that compiler does not optimize these calls away as retVal is not used anywhere...
   volatile typename StructType::TYPE retVal;

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = ((actDimens != dimens)? td->problemSize[0] : td->problemSize[i]);
      vecArr[i].resize(sizes[i]);
   }

   double commCost[MAX_EXEC_PLANS];
#ifdef SKEPU_CUDA
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   commCost[0] = 0.0;
   
   // the user can specify flag hints for operands memory location
   bool singlePlan = (td->extra->memUp != NULL);
   if(singlePlan)
   {
      assert(nImpls == 1);
      
      int *memUpFlags = td->extra->memUp;
      for(unsigned int i=0; i<actDimens; ++i)
      {
         if(memUpFlags[i] == 1) 
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
   }
   else
   {   
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
      }
   }
#else
   /*! if CUDA is not enabled we cannot consider the other cases */
   commCost[0] = 0.0;
   assert(nImpls == 1);
#endif

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::Reduce<StructType> redTest(userFunc);

   timer.start();

   retVal = redTest.OMP(vecArr[0]);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*OpenMP* reduce size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for MapOverlap skeleton and parallel OpenMP implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void omp_tune_wrapper_mapoverlap(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens >=1 && actDimens <= 2);

   // to ensure that compiler does not optimize these calls away as retVal is not used anywhere...
   volatile typename StructType::TYPE retVal;

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = ((actDimens != dimens)? td->problemSize[0] : td->problemSize[i]);
      vecArr[i].resize(sizes[i]);
   }

   double commCost[MAX_EXEC_PLANS];
#ifdef SKEPU_CUDA
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   commCost[0] = 0.0;
   
   // the user can specify flag hints for operands memory location
   bool singlePlan = (td->extra->memUp != NULL && td->extra->memDown != NULL);
   if(singlePlan)
   {
      assert(nImpls == 1);
      
      int *memUpFlags = td->extra->memUp;
      int *memDownFlags = td->extra->memDown;
      for(unsigned int i=0; i<actDimens; ++i)
      {
         if(i < (actDimens - 1) && memUpFlags[i] == 1) 
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
         else if(i == (actDimens - 1) && memDownFlags[0] == 1) 
            commCost[0] += bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
   }
   else
   {
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
      }
   }
#else
   /*! if CUDA is not enabled we cannot consider the other cases */
   commCost[0] = 0.0;
   assert(nImpls == 1);
#endif

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::MapOverlap<StructType> mapOverTest(userFunc);

   timer.start();

   if(actDimens == 1)
      mapOverTest.OMP(vecArr[0]);
   else if(actDimens == 2)
      mapOverTest.OMP(vecArr[0], vecArr[1]);
   else
      assert(false);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*OpenMP* mapoverlap size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for MapArray skeleton and parallel OpenMP implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void omp_tune_wrapper_maparray(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens >= 1 && dimens <= 2 && actDimens == 3);

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];

   sizes[0] = td->problemSize[0];
   sizes[1] = (dimens == 1)? td->problemSize[0] : td->problemSize[1];
   sizes[2] = (dimens == 1)? td->problemSize[0] : td->problemSize[1];

   for(unsigned int i=0; i<actDimens; ++i)
   {
      vecArr[i].resize(sizes[i]);
   }

   double commCost[MAX_EXEC_PLANS];
#ifdef SKEPU_CUDA
   assert(sizes[0] == sizes[1] && sizes[1] == sizes[2]);
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   commCost[0] = 0.0;
   
   // the user can specify flag hints for operands memory location
   bool singlePlan = (td->extra->memUp != NULL && td->extra->memDown != NULL);
   if(singlePlan)
   {
      assert(nImpls == 1);
      
      int *memUpFlags = td->extra->memUp;
      int *memDownFlags = td->extra->memDown;
      for(unsigned int i=0; i<actDimens; ++i)
      {
         if(i < (actDimens - 1) && memUpFlags[i] == 1) 
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
         else if(i == (actDimens - 1) && memDownFlags[0] == 1) 
            commCost[0] += bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
   }
   else
   {
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
      if(nImpls > 3)
      {
         assert(sizes[0] == sizes[1]); /*! TODO: fix it in future */
         commCost[3] = 0.0;
         commCost[4] = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[1]));
         commCost[5] = commCost[2] + bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[1]));
      }
   }
#else
   /*! if CUDA is not enabled we cannot consider the other cases */
   commCost[0] = 0.0;
   assert(nImpls == 1);
#endif

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::MapArray<StructType> mapArrTest(userFunc);

   timer.start();

   mapArrTest.OMP(vecArr[0], vecArr[1], vecArr[2]);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*OpenMP* maparray size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}


/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for MapReduce skeleton and parallel OpenMP implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void omp_tune_wrapper_mapreduce(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens >= 1 && actDimens <= 3);

   // to ensure that compiler does not optimize these calls away as retVal is not used anywhere...
   volatile typename StructType::TYPE retVal;

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = td->problemSize[0];
      vecArr[i].resize(sizes[i]);
   }

   double commCost[MAX_EXEC_PLANS];
#ifdef SKEPU_CUDA
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   commCost[0] = 0.0;
   double commCostPerOp = bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
   
   // the user can specify flag hints for operands memory location
   bool singlePlan = (td->extra->memUp != NULL);
   if(singlePlan)
   {
      assert(nImpls == 1);
      
      int *memUpFlags = td->extra->memUp;
      for(unsigned int i=0; i<actDimens; ++i)
      {
         if(memUpFlags[i] == 1) 
            commCost[0] += commCostPerOp;
      }
   }
   else
   {
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = commCostPerOp;
      }
      if(nImpls > 3)
      {
         commCost[3] = 0.0;
         commCost[4] = commCostPerOp;
         commCost[5] = commCostPerOp * 2;
      }
      if(nImpls > 6)
      {
         commCost[6] = 0.0;
         commCost[7] = commCostPerOp;
         commCost[8] = commCostPerOp * 2;
         commCost[9] = commCostPerOp * 3;
      }
   }
#else
   /*! if CUDA is not enabled we cannot consider the other cases */
   commCost[0] = 0.0;
   assert(nImpls == 1);
#endif

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   StructType2 *userFunc2 = new StructType2;
   if(td->callBackFunctionMapReduce != NULL)
      td->callBackFunctionMapReduce(userFunc, userFunc2, sizes, actDimens);

   skepu::MapReduce<StructType, StructType2> mapRedTest(userFunc, userFunc2);

   timer.start();

   if(actDimens == 1)
      retVal = mapRedTest.OMP(vecArr[0]);
   else if(actDimens == 2)
      retVal = mapRedTest.OMP(vecArr[0],vecArr[1]);
   else if(actDimens == 3)
      retVal = mapRedTest.OMP(vecArr[0],vecArr[1], vecArr[2]);
   else
      assert(false);

   timer.stop();

   DEBUG_TUNING_LEVEL3("*OpenMP* mapreduce size: " << sizes[0] << "\n");
   
   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}
#endif


/// the following functions train for CUDA implementations for different skeletons...

#ifdef SKEPU_CUDA

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for Map skeleton and CUDA implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void cuda_tune_wrapper_map(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;
   
   // the user can specify flag hints for operands memory location
   int *memUpFlags = td->extra->memUp;
   int *memDownFlags = td->extra->memDown;
   bool singlePlan = (memUpFlags != NULL && memDownFlags != NULL);

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens >= 1 && actDimens <= 4);

   cudaSetDevice(Environment<int>::getInstance()->bestCUDADevID);

   /// measuring communication cost...
   double commCost[MAX_EXEC_PLANS];
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   double commCostPerOp = bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
   
   
   if (singlePlan)
      commCost[0] = 0.0;
   
   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = td->problemSize[0];
      vecArr[i].resize(sizes[i]);

      if(singlePlan) // means data cost for HTD/DTH should be included assuming data is not already on required GPU memory
      {      
         if(i == (actDimens-1) && memDownFlags[0] == 0) 
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
         
         if(i < (actDimens-1) && memUpFlags[i] == 0) 
            commCost[0] += bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
      
      /*! transfer all operands to the memory... */
      if(i == actDimens-1) // last should be output variable and no copy is required in that case...
         vecArr[i].updateDevice_CU(&vecArr[i][0], sizes[i], Environment<int>::getInstance()->bestCUDADevID, false, true);
      else
         vecArr[i].updateDevice_CU(&vecArr[i][0], sizes[i], Environment<int>::getInstance()->bestCUDADevID, true, false);
   }

   cudaDeviceSynchronize();

   if(singlePlan)
      assert(nImpls == 1);
   else
   {
      commCost[0] = commCostPerOp * ((actDimens>1) ? (actDimens-1) : 1); // 0 operands are valid in gpu memory so need to transfer actDimens operands...
      if(nImpls > 1)
      {
         commCost[1] = commCostPerOp * ((actDimens>2) ? (actDimens-2) : 0);
         commCost[2] = commCostPerOp * ((actDimens>2) ? (actDimens-2) : 0);
      }
      if(nImpls > 3)
      {
         commCost[3] = commCostPerOp * ((actDimens>3) ? (actDimens-3) : 0);
         commCost[4] = commCostPerOp * ((actDimens>3) ? (actDimens-3) : 0);
         commCost[5] = commCostPerOp * ((actDimens>3) ? (actDimens-3) : 0);
      }
      if(nImpls > 6)
      {
         commCost[6] = 0;
         commCost[7] = 0;
         commCost[8] = 0;
         commCost[9] = 0;
      }
   }

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::Map<StructType> mapTest(userFunc);

   timer.start();

   if(actDimens == 1)
      mapTest.CU(vecArr[0]);
   else if(actDimens == 2)
      mapTest.CU(vecArr[0],vecArr[1]);
   else if(actDimens == 3)
      mapTest.CU(vecArr[0],vecArr[1], vecArr[2]);
   else if(actDimens == 4)
      mapTest.CU(vecArr[0],vecArr[1], vecArr[2], vecArr[3]);
   else
      assert(false);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*CUDA* map size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}


/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for Reduce skeleton and CUDA implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void cuda_tune_wrapper_reduce(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   // the user can specify flag hints for operands memory location
   int *memUpFlags = td->extra->memUp;
   bool singlePlan = (memUpFlags != NULL);

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens == 1);

   // to ensure that compiler does not optimize these calls away as retVal is not used anywhere...
   volatile typename StructType::TYPE retVal;

   /// measuring communication cost...
   double commCost[MAX_EXEC_PLANS];
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   
   if (singlePlan)
      commCost[0] = 0.0;
   
   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = td->problemSize[0];
      vecArr[i].resize(sizes[i]);

      if(singlePlan && memUpFlags[i] == 0) // means data cost for HTD should be included assuming data is not already on required GPU memory
      {      
         commCost[0] += bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
      
      /*! transfer all operand to the memory... */
      vecArr[i].updateDevice_CU(&vecArr[i][0], sizes[i], Environment<int>::getInstance()->bestCUDADevID, true, false);
   }

   cudaDeviceSynchronize();


   if(singlePlan)
      assert(nImpls == 1);
   else
   {
      commCost[0] = bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = 0.0;
      }
   }

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::Reduce<StructType> redTest(userFunc);

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   timer.start();

   retVal = redTest.CU(vecArr[0]); // DTH cost is implicit always in reduce and mapreduce patterns....

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*CUDA* reduce size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}


/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for MapOverlap skeleton and CUDA implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void cuda_tune_wrapper_mapoverlap(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;
   
   // the user can specify flag hints for operands memory location
   int *memUpFlags = td->extra->memUp;
   int *memDownFlags = td->extra->memDown;
   bool singlePlan = (memUpFlags != NULL && memDownFlags != NULL);

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens >=1 && actDimens <= 2);

   // to ensure that compiler does not optimize these calls away as retVal is not used anywhere...
   volatile typename StructType::TYPE retVal;

   /// measuring communication cost...
   double commCost[MAX_EXEC_PLANS];
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   
   if (singlePlan)
      commCost[0] = 0.0;
   
   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = td->problemSize[0];
      vecArr[i].resize(sizes[i]);

      if(singlePlan) // means data cost for HTD/DTH should be included assuming data is not already on required GPU memory
      {      
         if(i == 1 && memDownFlags[0] == 0)
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
         else if (i<1 && memUpFlags[i] == 0)
            commCost[0] += bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
      }
      
      /*! transfer all operand to the memory... */
      if(i == 1)
         vecArr[i].updateDevice_CU(&vecArr[i][0], sizes[i], Environment<int>::getInstance()->bestCUDADevID, false, true);
      else
         vecArr[i].updateDevice_CU(&vecArr[i][0], sizes[i], Environment<int>::getInstance()->bestCUDADevID, true, false);
   }

   cudaDeviceSynchronize();

   if(singlePlan)
      assert(nImpls == 1);
   else
   {
      commCost[0] = bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
      if(nImpls > 1)
      {
         commCost[1] = 0.0;
         commCost[2] = 0.0;
      }
   }

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::MapOverlap<StructType> mapOverTest(userFunc);

   timer.start();

   if(actDimens == 1)
      mapOverTest.CU(vecArr[0]);
   else if(actDimens == 2)
      mapOverTest.CU(vecArr[0], vecArr[1]);
   else
      assert(false);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*CUDA* mapoverlap size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for MapArray skeleton and CUDA implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void cuda_tune_wrapper_maparray(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;

   // the user can specify flag hints for operands memory location
   int *memUpFlags = td->extra->memUp;
   int *memDownFlags = td->extra->memDown;
   bool singlePlan = (memUpFlags != NULL && memDownFlags != NULL);


   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens >= 1 && dimens <= 2 && actDimens == 3);

   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];

   sizes[0] = td->problemSize[0];
   sizes[1] = (dimens == 1)? td->problemSize[0] : td->problemSize[1];
   sizes[2] = (dimens == 1)? td->problemSize[0] : td->problemSize[1];

   /// measuring communication cost...
   double commCost[MAX_EXEC_PLANS];
   assert(sizes[0] == sizes[1] && sizes[1] == sizes[2]);
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   
   if (singlePlan)
      commCost[0] = 0.0;
   
   for(unsigned int i=0; i<actDimens; ++i)
   {
      vecArr[i].resize(sizes[i]);

      if(singlePlan) // means data cost for HTD should be included assuming data is not already on required GPU memory
      {      
         if(i == (actDimens - 1) && memDownFlags[0] == 0)
            commCost[0] += bwDataStruct.latency_dth + (bwDataStruct.timing_dth * sizeof(typename StructType::TYPE) * (sizes[0]));
         else if (i < (actDimens - 1) && memUpFlags[i] == 0)
            commCost[0] += bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
      } 
      
      /*! transfer all operand to the memory... */
      if(i == actDimens-1) // last should be output variable and no copy is required in that case...
         vecArr[i].updateDevice_CU(&vecArr[i][0], sizes[i], Environment<int>::getInstance()->bestCUDADevID, false, true);
      else
         vecArr[i].updateDevice_CU(&vecArr[i][0], sizes[i], Environment<int>::getInstance()->bestCUDADevID, true, false);
   }

   cudaDeviceSynchronize();

   if(singlePlan)
      assert(nImpls == 1);
   else
   {
      commCost[0] = 2 * (bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0])));
      if(nImpls > 1)
      {
         commCost[1] = bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (sizes[0]));
         commCost[2] = commCost[1];
      }
      if(nImpls > 3)
      {
         commCost[3] = 0.0;
         commCost[4] = 0.0;
         commCost[5] = 0.0;
      }
   }

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   if(td->callBackFunction != NULL)
      td->callBackFunction(userFunc, sizes, actDimens);

   skepu::MapArray<StructType> mapArrTest(userFunc);

   timer.start();

   mapArrTest.CU(vecArr[0], vecArr[1], vecArr[2]);

   timer.stop();
   
   DEBUG_TUNING_LEVEL3("*CUDA* maparray size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}


/*!
 *  \ingroup tuning
 */
/*!
 * \brief Do training execution for a single performance context for MapReduce skeleton and CUDA implementation
 * \param arg a structure that includes information about performance context to train.
 */
template <typename StructType, typename StructType2>
void cuda_tune_wrapper_mapreduce(void *arg)
{
   if(!arg)
      return;

   Timer timer;

   TrainingData *td=reinterpret_cast<TrainingData*>(arg);
   assert(td != NULL);

   unsigned int nImpls = td->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   unsigned int dimens = td->dimens;
   unsigned int actDimens = td->extra->actDimensions;
   
   // the user can specify flag hints for operands memory location
   int *memUpFlags = td->extra->memUp;
   bool singlePlan = (memUpFlags != NULL);

   DEBUG_TUNING_LEVEL3("Computed dimensions: " << dimens << ", Actual dimensions: " << actDimens << "\n");

   assert(dimens == 1 && actDimens >= 1 && actDimens <= 3);

   // to ensure that compiler does not optimize these calls away as retVal is not used anywhere...
   volatile typename StructType::TYPE retVal;

   /// measuring communication cost...
   double commCost[MAX_EXEC_PLANS];
   DevTimingStruct &bwDataStruct = Environment<int>::getInstance()->bwDataStruct;
   double costPerOp = bwDataStruct.latency_htd + (bwDataStruct.timing_htd * sizeof(typename StructType::TYPE) * (td->problemSize[0]));
   
   if (singlePlan)
      commCost[0] = 0.0;
   
   size_t sizes[MAX_PARAMS];
   skepu::Vector<typename StructType::TYPE> vecArr[MAX_PARAMS];
   for(unsigned int i=0; i<actDimens; ++i)
   {
      sizes[i] = td->problemSize[0];
      vecArr[i].resize(sizes[i]);
      
      if(singlePlan && memUpFlags[i] == 0) // means data cost for HTD should be included assuming data is not already on required GPU memory
      {
         commCost[0] += costPerOp;
      }

      /*! transfer all operand to the memory... */
      vecArr[i].updateDevice_CU(&vecArr[i][0], sizes[i], Environment<int>::getInstance()->bestCUDADevID, true, false);
   }

   cudaDeviceSynchronize();
  
   if(singlePlan)
      assert(nImpls == 1);
   else
   {
      commCost[0] = costPerOp * actDimens; // 0 operands are valid in gpu memory so need to transfer actDimens operands...
      if(nImpls > 1)
      {
         commCost[1] = costPerOp * ((actDimens>1) ? (actDimens-1) : 0);
         commCost[2] = costPerOp * ((actDimens>1) ? (actDimens-1) : 0);
      }
      if(nImpls > 3)
      {
         commCost[3] = costPerOp * ((actDimens>2) ? (actDimens-2) : 0);
         commCost[4] = costPerOp * ((actDimens>2) ? (actDimens-2) : 0);
         commCost[5] = costPerOp * ((actDimens>2) ? (actDimens-2) : 0);
      }
      if(nImpls > 6)
      {
         commCost[6] = 0;
         commCost[7] = 0;
         commCost[8] = 0;
         commCost[9] = 0;
      }
   }

   /*! to allow user control e.g. setting constant value etc.. */
   StructType *userFunc = new StructType;
   StructType2 *userFunc2 = new StructType2;
   if(td->callBackFunctionMapReduce != NULL)
      td->callBackFunctionMapReduce(userFunc, userFunc2, sizes, actDimens);

   skepu::MapReduce<StructType, StructType2> mapRedTest(userFunc, userFunc2);

   timer.start();

   if(actDimens == 1)
      retVal = mapRedTest.CU(vecArr[0]);
   else if(actDimens == 2)
      retVal = mapRedTest.CU(vecArr[0],vecArr[1]);
   else if(actDimens == 3)
      retVal = mapRedTest.CU(vecArr[0],vecArr[1], vecArr[2]);
   else
      assert(false);

   timer.stop(); // DTH cost is implicit always in reduce and mapreduce patterns....
   
   DEBUG_TUNING_LEVEL3("*CUDA* mapreduce size: " << sizes[0] << "\n");

   std::string printStr = "";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      td->exec_time[i] = commCost[i] + timer.getTotalTime();
      printStr += " " + convertToStr<double>(td->exec_time[i]);
   }
   DEBUG_TUNING_LEVEL3(printStr + "\n");
}
#endif








/*!
 *  \ingroup tuning
 */
/*!
 * \brief Tuner class: generic definition....
 * Multiple class specializations are defined for this class, one for each skeleton type. 
 * It allows to avoid possible compiler errors considering differences in function arguments for different skeleton types. 
 */
template <typename StructType, SkeletonType type, typename StructType2 = StructType>
struct Tuner
{
   Tuner()
   {
      assert(false);
   }
};


/*!
 *  \ingroup tuning
 */
/*!
 * \brief A helper function that creates the default configuration
 * \param bp a structure that is written with default settings enabled in SkePU
 */
void createDefaultConfiguration(BackEndParams &bp)
{
   Environment<int> *environment = Environment<int>::getInstance();

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
   bp.maxThreads = environment->m_devices_CL.at(0)->getMaxThreads();
   bp.maxBlocks = environment->m_devices_CL.at(0)->getMaxBlocks();
#endif

#ifdef SKEPU_CUDA
   bp.maxThreads = environment->m_devices_CU.at(0)->getMaxThreads();
   bp.maxBlocks = environment->m_devices_CU.at(0)->getMaxBlocks();
#endif

#ifdef SKEPU_OPENMP
#ifdef SKEPU_OPENMP_THREADS
   bp.numOmpThreads = SKEPU_OPENMP_THREADS;
#else
   bp.numOmpThreads = omp_get_max_threads();
#endif
#endif
}





/*! This functionality only for non-windows platform */
#ifndef _WIN32

#include "skepu/src/makedir.h"


/*!
 *  \ingroup tuning
 */
/*!
 * \brief Loads an execution plan for a file into the structure passes as argument.
 * \param id The identifier for skeletonlet being tuned (skeleton type + user function). It defines the filename.
 * \param plan The execution plan where plan is loaded.
 */
bool loadExecPlan(std::string id, ExecPlan &plan)
{
   std::string path = getPMDirectory();
   path += id + ".meta";
   if(fileExists(path))
   {
      std::ifstream infile(path.c_str());

      assert(infile.good());

      std::string strLine;
      size_t low, upp, numCUThreads, numCUBlocks;
      unsigned int numOmpThreads;

      std::string impTypeStr;
      BackEndParams bp;
      while(infile.good())
      {
         getline(infile, strLine);
         strLine = trimSpaces(strLine);
         if(strLine[0] == '%' || strLine[0] == '/' || strLine[0] == '#')
            continue;

         std::istringstream iss(strLine);
         iss >> low >> upp >> impTypeStr;
         iss >> numOmpThreads >> numCUThreads >> numCUBlocks;

         bp.numOmpThreads = numOmpThreads;
         bp.maxThreads = numCUThreads;
         bp.maxBlocks = numCUBlocks;

         impTypeStr = capitalizeString(impTypeStr);

         if(impTypeStr == "CPU")
         {
            bp.backend = CPU_BACKEND;

         }
         else if(impTypeStr == "OMP")
         {
            bp.backend = OMP_BACKEND;
//              iss >> numOmpThreads;
         }
         else if(impTypeStr == "CUDA")
         {
            bp.backend = CU_BACKEND;
//              iss >> numCUThreads >> numCUBlocks;
         }
         else
            assert(false);

         plan.add(low, upp, bp);
      }
      plan.calibrated = true;
      return true;
   }
   return false;
}


/*!
 *  \ingroup tuning
 */
/*!
 * \brief Stores an execution plan for the structure passed as argument to a file.
 * \param id The identifier for skeletonlet being tuned (skeleton type + user function). It defines the filename.
 * \param planArray The execution plan from where plan is stored.
 */
bool storeExecPlan(std::string id, const ExecPlan &plan)
{
   std::string path = getPMDirectory();
   std::string file(path + id + ".meta");

   if(fileExists(file) == false)
      createPath(path);


   std::ofstream outfile(file.c_str());

   assert(outfile.good());


   outfile << "% Execution plan for " << id << "\n";
   std::map< std::pair<size_t, size_t>, BackEndParams > m_data = plan.sizePlan;
   for(std::map< std::pair<size_t, size_t>, BackEndParams >::iterator it = m_data.begin(); it != m_data.end(); ++it)
   {
      std::string beTypeStr = "";
      BackEndParams bp = it->second;
      switch(bp.backend)
      {
      case CPU_BACKEND:
         beTypeStr = "CPU " + convertIntToString(bp.numOmpThreads) + " " + convertIntToString(bp.maxThreads) + " " + convertIntToString(bp.maxBlocks);
         break;
      case OMP_BACKEND:
         beTypeStr = "OMP " + convertIntToString(bp.numOmpThreads) + " " + convertIntToString(bp.maxThreads) + " " + convertIntToString(bp.maxBlocks);
         break;
      case CU_BACKEND:
         beTypeStr = "CUDA " + convertIntToString(bp.numOmpThreads) + " " + convertIntToString(bp.maxThreads) + " " + convertIntToString(bp.maxBlocks);
         break;
      default:
         assert(false);
      }

      outfile << it->first.first << " " << it->first.second << " " << beTypeStr << "\n";
   }

   outfile.close();

   return true;
}


/*!
 *  \ingroup tuning
 */
/*!
 * \brief Loads execution plans for a file into the structure passes as argument.
 * \param id The identifier for skeletonlet being tuned (skeleton type + user function). It defines the filename.
 * \param planArray The execution plan array where plans are loaded.
 */
bool loadExecPlanArray(std::string id, ExecPlan *planArray)
{
   assert(planArray != NULL);

   std::string path = getPMDirectory();
   path += id + "_multi.meta";
   if(fileExists(path))
   {
      std::ifstream infile(path.c_str());

      assert(infile.good());

      std::string strLine;
      size_t low, upp, numCUThreads, numCUBlocks;
      unsigned int numOmpThreads;
      std::string impTypeStr;
//       ImplType impType;
      BackEndParams bp;
      int idx = -1;
      while(infile.good())
      {
         getline(infile, strLine);
         strLine = trimSpaces(strLine);
         if(strLine[0] == '%' || strLine[0] == '/')
            continue;

         if(strLine[0] == '#')
         {
            strLine = trimSpaces(strLine.substr(1));
            int tmpIdx;
            std::istringstream iss(strLine);
            iss >> tmpIdx;
            idx++;
            assert(idx == tmpIdx);
            continue;
         }
         assert(idx >= 0 && idx < MAX_EXEC_PLANS);

         std::istringstream iss(strLine);
         iss >> low >> upp >> impTypeStr;
         iss >> numOmpThreads >> numCUThreads >> numCUBlocks;

         bp.numOmpThreads = numOmpThreads;
         bp.maxThreads = numCUThreads;
         bp.maxBlocks = numCUBlocks;

         impTypeStr = capitalizeString(impTypeStr);

         if(impTypeStr == "CPU")
         {
            bp.backend = CPU_BACKEND;

         }
         else if(impTypeStr == "OMP")
         {
            bp.backend = OMP_BACKEND;
// 		iss >> numOmpThreads;
         }
         else if(impTypeStr == "CUDA")
         {
            bp.backend = CU_BACKEND;
// 		iss >> numCUThreads >> numCUBlocks;
         }
         else
            assert(false);

         planArray[idx].add(low, upp, bp);
         planArray[idx].calibrated = true;
      }
      return true;
   }
   return false;
}

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Stores execution plans for the structure passed as argument to a file.
 * \param id The identifier for skeletonlet being tuned (skeleton type + user function). It defines the filename.
 * \param planArray The execution plan array from where plans are stored.
 */
bool storeExecPlanArray(std::string id, const ExecPlan *planArray, unsigned int nImpls)
{
   assert(planArray != NULL);

   std::string path = getPMDirectory();
   std::string file(path + id + "_multi.meta");

   if(fileExists(file) == false)
      createPath(path);


   std::ofstream outfile(file.c_str());

   assert(outfile.good());

   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);

   outfile << "% Execution plan for " << id << "\n";
   for(unsigned int i=0; i<nImpls; ++i)
   {
      if(!planArray[i].calibrated)
      {
         SKEPU_WARNING("[SKEPU Warning]: Plan '" << id << "' is not calibrated for index: " << i << "\n");
         break;
      }

      outfile << "# " << i << "\n";

      const std::map< std::pair<size_t, size_t>, BackEndParams > &m_data = planArray[i].sizePlan;
      for(std::map< std::pair<size_t, size_t>, BackEndParams >::const_iterator it = m_data.begin(); it != m_data.end(); ++it)
      {
         std::string beTypeStr = "";
         BackEndParams bp = it->second;
         switch(bp.backend)
         {
         case CPU_BACKEND:
            beTypeStr = "CPU " + convertIntToString(bp.numOmpThreads) + " " + convertIntToString(bp.maxThreads) + " " + convertIntToString(bp.maxBlocks);
            break;
         case OMP_BACKEND:
            beTypeStr = "OMP " + convertIntToString(bp.numOmpThreads) + " " + convertIntToString(bp.maxThreads) + " " + convertIntToString(bp.maxBlocks);
            break;
         case CU_BACKEND:
            beTypeStr = "CUDA " + convertIntToString(bp.numOmpThreads) + " " + convertIntToString(bp.maxThreads) + " " + convertIntToString(bp.maxBlocks);
            break;
         default:
            assert(false);
         }

         outfile << it->first.first << " " << it->first.second << " " << beTypeStr << "\n";
      }
   }

   outfile.close();

   return true;
}

#endif

/*!
 *  \ingroup tuning
 */
/*!
 * \brief Tuner class specilization for MapReduce skeleton
 */
template <typename StructType, typename StructType2>
struct Tuner<StructType, MAPREDUCE, StructType2>
{
   Tuner(std::string _id, int _dimens, size_t *_lowBounds, size_t *_uppBounds): id(_id), dimens(_dimens), lowBounds(_lowBounds), uppBounds(_uppBounds), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(dimens >= 1 && dimens <= 3 && lowBounds && uppBounds);
      extra.memUp = NULL;
      extra.memDown = NULL;
   }

   Tuner(std::string _id, int _dimens, size_t *_lowBounds, size_t *_uppBounds, int *_memUp): id(_id), dimens(_dimens), lowBounds(_lowBounds), uppBounds(_uppBounds), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(dimens >= 1 && dimens <= 3 && lowBounds && uppBounds);
      extra.memUp = _memUp;
      extra.memDown = NULL;
   }

   StatsTuner stats;

   void operator()(ExecPlan *execPlanArray)
   {
      assert(execPlanArray!=NULL);

      /*!
       * Internal logic uses memup and memdown variable to find out whether to use hints or the second mode
       * memup and memdown should be null to make sure that they are not accessed
       */
      int *oldMemUp = extra.memUp;
      extra.memUp = NULL;
      
      unsigned int actDimens = dimens;
      std::string interface = "mapreduce";
      dimens = 1;

      /*! TODO: for simplification, fix later... */
      unsigned int nImpls = 1;
   #ifdef SKEPU_CUDA
         nImpls = nImpls = ( (actDimens == 1) ? 3 : ((actDimens == 2) ? 6 : 10) );
   #endif
      assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);

      /*! This functionality only for non-windows platform */
#if !defined(_WIN32) && !defined(REDO_MEASUREMENTS)
      if(loadExecPlanArray(id, execPlanArray))
      {
         bool redoMesures = false;
         for(unsigned int i=0; i<nImpls; ++i)
         {
            if(execPlanArray[i].calibrated == false)
            {
               redoMesures = true;
               break;
            }
            for(unsigned int i=0; i<dimens; ++i)
            {
               if(execPlanArray[i].isTrainedFor(lowBounds[i]) == false || execPlanArray[i].isTrainedFor(uppBounds[i]) == false)
               {
                  redoMesures = true;
                  break;
               }
            }
         }
         /*! only use existing stored ExecPlan when it has same no of Impls and also trained (atleast) for the required input range  */
         if(redoMesures == false)
            return;
      }
#endif

      BackEndParams bp;
      createDefaultConfiguration(bp);

      std::vector<size_t> upperBounds(dimens);
      std::vector<size_t> lowerBounds(dimens);

      for(unsigned int i=0; i<dimens; ++i)
      {
         upperBounds[i] = uppBounds[i];
         lowerBounds[i] = lowBounds[i];
      }

      std::vector<ImpDetail*> impls;

      cpu_tune_wrapper_mapreduce<StructType, StructType2>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cpu_impl",  IMPL_CPU,   &cpu_tune_wrapper_mapreduce<StructType, StructType2>));

#ifdef SKEPU_OPENMP
      omp_tune_wrapper_mapreduce<StructType, StructType2>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("omp_impl",  IMPL_OMP,   &omp_tune_wrapper_mapreduce<StructType, StructType2>));
#endif

#ifdef SKEPU_CUDA
      cuda_tune_wrapper_mapreduce<StructType, StructType2>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cuda_impl",  IMPL_CUDA,   &cuda_tune_wrapper_mapreduce<StructType, StructType2>));
#endif

      std::ofstream outfile(std::string("tree_data_multi_" + id + ".dat").c_str());
      assert(outfile.good());


      extra.actDimensions = actDimens;
      Trainer trainer(impls, lowerBounds, upperBounds, MAX_DEPTH, nImpls, extra, callBackFunction, callBackFunctionMapReduce, OVERSAMPLE);
      trainer.train();

      ExecPlanNew<1> planArr[MAX_EXEC_PLANS];
      trainer.constructExecPlanNew(planArr, stats);

      for(unsigned int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         if(planArr[i].calibrated == false)
            break;

         outfile << planArr[i];
      }

      for(unsigned int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         if(planArr[i].calibrated == false)
            break;

         execPlanArray[i].clear();
         outfile << "compressed plan:\n";
         trainer.compressExecPlanNew(planArr[i]);
         for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = planArr[i].m_data.begin(); it != planArr[i].m_data.end(); ++it)
         {
            switch(it->second)
            {
            case IMPL_CPU:
               bp.backend = CPU_BACKEND;
               break;
            case IMPL_OMP:
               bp.backend = OMP_BACKEND;
               break;
            case IMPL_CUDA:
               bp.backend = CU_BACKEND;
               break;
            default:
               assert(false);
            }
            execPlanArray[i].add(it->first.first, it->first.second, bp);
            execPlanArray[i].calibrated = true;
         }
         outfile << planArr[i];
      }

      /*! This functionality only for non-windows platform */
#ifndef _WIN32
      storeExecPlanArray(id, execPlanArray, nImpls);
#endif

      outfile << *(trainer.m_tree);
      DEBUG_TUNING_LEVEL2( "\nTree: " << *(trainer.m_tree) << "\n");

      // free memory...
      for(unsigned int i=0; i<impls.size(); ++i)
      {
         delete impls[i];
      }

      impls.clear();
      
      // restore them now...
      extra.memUp = oldMemUp;
   }
   
   
   ExecPlan operator()()
   {
      assert(extra.memUp != NULL);

      unsigned int actDimens = dimens;
      std::string interface = "mapreduce";
      dimens = 1;
      
      unsigned int nImpls = 1;

     ExecPlan execPlan;

      /*! This functionality only for non-windows platform */
#if !defined(_WIN32) && !defined(REDO_MEASUREMENTS)
      if(loadExecPlan(id, execPlan))
      {
         return execPlan;
      }
#endif

      BackEndParams bp;
      createDefaultConfiguration(bp);

      std::vector<size_t> upperBounds(dimens);
      std::vector<size_t> lowerBounds(dimens);

      for(unsigned int i=0; i<dimens; ++i)
      {
         upperBounds[i] = uppBounds[i];
         lowerBounds[i] = lowBounds[i];
      }

      std::vector<ImpDetail*> impls;

      cpu_tune_wrapper_mapreduce<StructType, StructType2>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cpu_impl",  IMPL_CPU,   &cpu_tune_wrapper_mapreduce<StructType, StructType2>));

#ifdef SKEPU_OPENMP
      omp_tune_wrapper_mapreduce<StructType, StructType2>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("omp_impl",  IMPL_OMP,   &omp_tune_wrapper_mapreduce<StructType, StructType2>));
#endif

#ifdef SKEPU_CUDA
      cuda_tune_wrapper_mapreduce<StructType, StructType2>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cuda_impl",  IMPL_CUDA,   &cuda_tune_wrapper_mapreduce<StructType, StructType2>));
#endif

      std::ofstream outfile(std::string("tree_data_" + id + ".dat").c_str());
      assert(outfile.good());

      extra.actDimensions = actDimens;
      Trainer trainer(impls, lowerBounds, upperBounds, MAX_DEPTH, nImpls, extra, callBackFunction, callBackFunctionMapReduce, OVERSAMPLE);
      trainer.train();

      ExecPlanNew<1> plan;
      trainer.constructExecPlanNew(&plan, stats);
      assert(plan.calibrated);
      outfile << plan ;

      outfile << "compressed plan:\n";
      trainer.compressExecPlanNew(plan);
      for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = plan.m_data.begin(); it != plan.m_data.end(); ++it)
      {
         switch(it->second)
         {
         case IMPL_CPU:
            bp.backend = CPU_BACKEND;
            break;
         case IMPL_OMP:
            bp.backend = OMP_BACKEND;
            break;
         case IMPL_CUDA:
            bp.backend = CU_BACKEND;
            break;
         default:
            assert(false);
         }
         execPlan.add(it->first.first, it->first.second, bp);
      }
      outfile << plan;

      /*! This functionality only for non-windows platform */
#ifndef _WIN32
      storeExecPlan(id, execPlan);
#endif

      outfile << *(trainer.m_tree);
      DEBUG_TUNING_LEVEL2( "\nTree: " << *(trainer.m_tree) << "\n");
      
      // free memory...
      for(int i=0; i<impls.size(); ++i)
      {
         delete impls[i];
      }

      impls.clear();

      return execPlan;
   }

public:
   void (*callBackFunction)(void*, size_t*, unsigned int);
   void (*callBackFunctionMapReduce)(void*, void*, size_t*, unsigned int);

private:
   ExtraData extra;
   unsigned int dimens;
   size_t *lowBounds;
   size_t *uppBounds;
   std::string id;
};


/*!
 *  \ingroup tuning
 */
/*!
 * \brief Tuner class specilization for Map skeleton
 */
template <typename StructType>
struct Tuner<StructType, MAP, StructType>
{
   Tuner(std::string _id, int _dimens, size_t *_lowBounds, size_t *_uppBounds): id(_id), dimens(_dimens), lowBounds(_lowBounds), uppBounds(_uppBounds), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(dimens >= 1 && dimens <= 4 && lowBounds && uppBounds);
      extra.memUp = NULL;
      extra.memDown = NULL;
   }

   Tuner(std::string _id, int _dimens, size_t *_lowBounds, size_t *_uppBounds, int *_memUp, int *_memDown): id(_id), dimens(_dimens), lowBounds(_lowBounds), uppBounds(_uppBounds), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(dimens >= 1 && dimens <= 4 && lowBounds && uppBounds);
      extra.memUp = _memUp;
      extra.memDown = _memDown;
   }

   StatsTuner stats;

   void operator()(ExecPlan *execPlanArray)
   {
      assert(execPlanArray!=NULL);

      /*!
       * Internal logic uses memup and memdown variable to find out whether to use hints or the second mode
       * memup and memdown should be null to make sure that they are not accessed
       */
      int *oldMemUp = extra.memUp;
      extra.memUp = NULL;
      int *oldMemDown = extra.memDown;
      extra.memDown = NULL;
      
      
      unsigned int actDimens = dimens;
      std::string interface = "map";
      dimens = 1;

      /*! TODO: for simplification, fix later... */
      unsigned int nImpls = 1;
   #ifdef SKEPU_CUDA
         nImpls = ( (actDimens == 1 || actDimens == 2) ? 3 : ((actDimens == 3) ? 6 : 10) );
   #endif
      assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);

      /*! This functionality only for non-windows platform */
#if !defined(_WIN32) && !defined(REDO_MEASUREMENTS)
      if(loadExecPlanArray(id, execPlanArray))
      {
         bool redoMesures = false;
         for(unsigned int i=0; i<nImpls; ++i)
         {
            if(execPlanArray[i].calibrated == false)
            {
               redoMesures = true;
               break;
            }
            for(unsigned int i=0; i<dimens; ++i)
            {
               if(execPlanArray[i].isTrainedFor(lowBounds[i]) == false || execPlanArray[i].isTrainedFor(uppBounds[i]) == false)
               {
                  redoMesures = true;
                  break;
               }
            }
         }
         /*! only use existing stored ExecPlan when it has same no of Impls and also trained (atleast) for the required input range  */
         if(redoMesures == false)
            return;
      }
#endif

      BackEndParams bp;
      createDefaultConfiguration(bp);

      std::vector<size_t> upperBounds(dimens);
      std::vector<size_t> lowerBounds(dimens);

      for(unsigned int i=0; i<dimens; ++i)
      {
         upperBounds[i] = uppBounds[i];
         lowerBounds[i] = lowBounds[i];
      }

      std::vector<ImpDetail*> impls;

      cpu_tune_wrapper_map<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cpu_impl",  IMPL_CPU,   &cpu_tune_wrapper_map<StructType, StructType>));

#ifdef SKEPU_OPENMP
      omp_tune_wrapper_map<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("omp_impl",  IMPL_OMP,   &omp_tune_wrapper_map<StructType, StructType>));
#endif

#ifdef SKEPU_CUDA
      cuda_tune_wrapper_map<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cuda_impl",  IMPL_CUDA,   &cuda_tune_wrapper_map<StructType, StructType>));
#endif

      std::ofstream outfile(std::string("tree_data_multi_" + id + ".dat").c_str());
      assert(outfile.good());

      extra.actDimensions = actDimens;
      Trainer trainer(impls, lowerBounds, upperBounds, MAX_DEPTH, nImpls, extra, callBackFunction, callBackFunctionMapReduce, OVERSAMPLE);
      trainer.train();

      ExecPlanNew<1> planArr[MAX_EXEC_PLANS];
      trainer.constructExecPlanNew(planArr, stats);

      for(unsigned int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         if(planArr[i].calibrated == false)
            break;

         outfile << planArr[i];
      }

      for(unsigned int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         if(planArr[i].calibrated == false)
            break;

         execPlanArray[i].clear();
         outfile << "compressed plan:\n";
         trainer.compressExecPlanNew(planArr[i]);
         for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = planArr[i].m_data.begin(); it != planArr[i].m_data.end(); ++it)
         {
            switch(it->second)
            {
            case IMPL_CPU:
               bp.backend = CPU_BACKEND;
               break;
            case IMPL_OMP:
               bp.backend = OMP_BACKEND;
               break;
            case IMPL_CUDA:
               bp.backend = CU_BACKEND;
               break;
            default:
               assert(false);
            }
            execPlanArray[i].add(it->first.first, it->first.second, bp);
            execPlanArray[i].calibrated = true;
         }
         outfile << planArr[i] ;
      }

      /*! This functionality only for non-windows platform */
#ifndef _WIN32
      storeExecPlanArray(id, execPlanArray, nImpls);
#endif

      outfile << *(trainer.m_tree);
      DEBUG_TUNING_LEVEL2( "\nTree: " << *(trainer.m_tree) << "\n");

      // free memory...
      for(unsigned int i=0; i<impls.size(); ++i)
      {
         delete impls[i];
      }

      impls.clear();
      
      // restore them now...
      extra.memUp = oldMemUp;
      extra.memDown = oldMemDown;
   }
   
   
   ExecPlan operator()()
   {
      assert(extra.memUp != NULL && extra.memDown != NULL);

      unsigned int actDimens = dimens;
      std::string interface = "map";
      dimens = 1;
      
      unsigned int nImpls = 1;

      ExecPlan execPlan;

      /*! This functionality only for non-windows platform */
#if !defined(_WIN32) && !defined(REDO_MEASUREMENTS)
      if(loadExecPlan(id, execPlan))
      {
         return execPlan;
      }
#endif

      BackEndParams bp;
      createDefaultConfiguration(bp);

      std::vector<size_t> upperBounds(dimens);
      std::vector<size_t> lowerBounds(dimens);

      for(unsigned int i=0; i<dimens; ++i)
      {
         upperBounds[i] = uppBounds[i];
         lowerBounds[i] = lowBounds[i];
      }

      std::vector<ImpDetail*> impls;

      cpu_tune_wrapper_map<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cpu_impl",  IMPL_CPU,   &cpu_tune_wrapper_map<StructType, StructType>));

#ifdef SKEPU_OPENMP
      omp_tune_wrapper_map<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("omp_impl",  IMPL_OMP,   &omp_tune_wrapper_map<StructType, StructType>));
#endif

#ifdef SKEPU_CUDA
      cuda_tune_wrapper_map<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cuda_impl",  IMPL_CUDA,   &cuda_tune_wrapper_map<StructType, StructType>));
#endif

      std::ofstream outfile(std::string("tree_data_" + id + ".dat").c_str());
      assert(outfile.good());

      extra.actDimensions = actDimens;
      Trainer trainer(impls, lowerBounds, upperBounds, MAX_DEPTH, nImpls, extra, callBackFunction, callBackFunctionMapReduce, OVERSAMPLE);
      trainer.train();

      ExecPlanNew<1> plan;
      trainer.constructExecPlanNew(&plan, stats);
      assert(plan.calibrated);
      outfile << plan ;

      outfile << "compressed plan:\n";
      trainer.compressExecPlanNew(plan);
      for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = plan.m_data.begin(); it != plan.m_data.end(); ++it)
      {
         switch(it->second)
         {
         case IMPL_CPU:
            bp.backend = CPU_BACKEND;
            break;
         case IMPL_OMP:
            bp.backend = OMP_BACKEND;
            break;
         case IMPL_CUDA:
            bp.backend = CU_BACKEND;
            break;
         default:
            assert(false);
         }
         execPlan.add(it->first.first, it->first.second, bp);
      }
      outfile << plan;

      /*! This functionality only for non-windows platform */
#ifndef _WIN32
      storeExecPlan(id, execPlan);
#endif

      outfile << *(trainer.m_tree);
      DEBUG_TUNING_LEVEL2( "\nTree: " << *(trainer.m_tree) << "\n");
      
      // free memory...
      for(int i=0; i<impls.size(); ++i)
      {
         delete impls[i];
      }

      impls.clear();

      return execPlan;
   }

public:
   void (*callBackFunction)(void*, size_t*, unsigned int);
   void (*callBackFunctionMapReduce)(void*, void*, size_t*, unsigned int);

private:
   ExtraData extra;
   unsigned int dimens;
   size_t *lowBounds;
   size_t *uppBounds;
   std::string id;
};





/*!
 *  \ingroup tuning
 */
/*!
 * \brief Tuner class specilization for Reduce skeleton
 */
template <typename StructType>
struct Tuner<StructType, REDUCE, StructType>
{
   Tuner(std::string _id, int _dimens, size_t *_lowBounds, size_t *_uppBounds): id(_id), dimens(_dimens), lowBounds(_lowBounds), uppBounds(_uppBounds), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(dimens == 1 && lowBounds && uppBounds);
      extra.memUp = NULL;
      extra.memDown = NULL;
   }

   Tuner(std::string _id, int _dimens, size_t *_lowBounds, size_t *_uppBounds, int *_memUp): id(_id), dimens(_dimens), lowBounds(_lowBounds), uppBounds(_uppBounds), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(dimens == 1 && lowBounds && uppBounds);
      extra.memUp = _memUp;
      extra.memDown = NULL;
   }

   StatsTuner stats;

   void operator()(ExecPlan *execPlanArray)
   {
      assert(execPlanArray!=NULL);

      /*!
       * Internal logic uses memup and memdown variable to find out whether to use hints or the second mode
       * memup and memdown should be null to make sure that they are not accessed
       */
      int *oldMemUp = extra.memUp;
      extra.memUp = NULL;

      unsigned int actDimens = dimens;
      std::string interface = "reduce";

      /*! TODO: for simplification, fix later... */
      unsigned int nImpls = 1;
   #ifdef SKEPU_CUDA
         nImpls = 3;
   #endif
      assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);

      /*! This functionality only for non-windows platform */
#if !defined(_WIN32) && !defined(REDO_MEASUREMENTS)
      if(loadExecPlanArray(id, execPlanArray))
      {
         bool redoMesures = false;
         for(unsigned int i=0; i<nImpls; ++i)
         {
            if(execPlanArray[i].calibrated == false)
            {
               redoMesures = true;
               break;
            }
            for(unsigned int i=0; i<dimens; ++i)
            {
               if(execPlanArray[i].isTrainedFor(lowBounds[i]) == false || execPlanArray[i].isTrainedFor(uppBounds[i]) == false)
               {
                  redoMesures = true;
                  break;
               }
            }
         }
         /*! only use existing stored ExecPlan when it has same no of Impls and also trained (atleast) for the required input range  */
         if(redoMesures == false)
            return;
      }
#endif

      BackEndParams bp;
      createDefaultConfiguration(bp);

      std::vector<size_t> upperBounds(dimens);
      std::vector<size_t> lowerBounds(dimens);

      for(unsigned int i=0; i<dimens; ++i)
      {
         upperBounds[i] = uppBounds[i];
         lowerBounds[i] = lowBounds[i];
      }

      std::vector<ImpDetail*> impls;

      cpu_tune_wrapper_reduce<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cpu_impl",  IMPL_CPU,   &cpu_tune_wrapper_reduce<StructType, StructType>));

#ifdef SKEPU_OPENMP
      omp_tune_wrapper_reduce<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("omp_impl",  IMPL_OMP,   &omp_tune_wrapper_reduce<StructType, StructType>));
#endif

#ifdef SKEPU_CUDA
      cuda_tune_wrapper_reduce<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cuda_impl",  IMPL_CUDA,   &cuda_tune_wrapper_reduce<StructType, StructType>));
#endif

      std::ofstream outfile(std::string("tree_data_multi_" + id + ".dat").c_str());
      assert(outfile.good());

      extra.actDimensions = actDimens;
      Trainer trainer(impls, lowerBounds, upperBounds, MAX_DEPTH, nImpls, extra, callBackFunction, callBackFunctionMapReduce, OVERSAMPLE);
      trainer.train();

      ExecPlanNew<1> planArr[MAX_EXEC_PLANS];
      trainer.constructExecPlanNew(planArr, stats);

      for(unsigned int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         if(planArr[i].calibrated == false)
            break;

         outfile << planArr[i];
      }

      for(unsigned int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         if(planArr[i].calibrated == false)
            break;

         execPlanArray[i].clear();
         outfile << "compressed plan:\n";
         trainer.compressExecPlanNew(planArr[i]);
         for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = planArr[i].m_data.begin(); it != planArr[i].m_data.end(); ++it)
         {
            switch(it->second)
            {
            case IMPL_CPU:
               bp.backend = CPU_BACKEND;
               break;
            case IMPL_OMP:
               bp.backend = OMP_BACKEND;
               break;
            case IMPL_CUDA:
               bp.backend = CU_BACKEND;
               break;
            default:
               assert(false);
            }
            execPlanArray[i].add(it->first.first, it->first.second, bp);
            execPlanArray[i].calibrated = true;
         }
         outfile << planArr[i];
      }

      /*! This functionality only for non-windows platform */
#ifndef _WIN32
      storeExecPlanArray(id, execPlanArray, nImpls);
#endif

      outfile << *(trainer.m_tree);
      DEBUG_TUNING_LEVEL2( "\nTree: " << *(trainer.m_tree) << "\n");

      // free memory...
      for(unsigned int i=0; i<impls.size(); ++i)
      {
         delete impls[i];
      }

      impls.clear();
      
      // restore them now...
      extra.memUp = oldMemUp;
   }
   
   
   ExecPlan operator()()
   {
      assert(extra.memUp != NULL);

      unsigned int actDimens = dimens;
      std::string interface = "map";
      dimens = 1;
      
      unsigned int nImpls = 1;

      ExecPlan execPlan;

      /*! This functionality only for non-windows platform */
#if !defined(_WIN32) && !defined(REDO_MEASUREMENTS)
      if(loadExecPlan(id, execPlan))
      {
         return execPlan;
      }
#endif

      BackEndParams bp;
      createDefaultConfiguration(bp);

      std::vector<size_t> upperBounds(dimens);
      std::vector<size_t> lowerBounds(dimens);

      for(unsigned int i=0; i<dimens; ++i)
      {
         upperBounds[i] = uppBounds[i];
         lowerBounds[i] = lowBounds[i];
      }

      std::vector<ImpDetail*> impls;

      cpu_tune_wrapper_reduce<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cpu_impl",  IMPL_CPU,   &cpu_tune_wrapper_reduce<StructType, StructType>));

#ifdef SKEPU_OPENMP
      omp_tune_wrapper_reduce<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("omp_impl",  IMPL_OMP,   &omp_tune_wrapper_reduce<StructType, StructType>));
#endif

#ifdef SKEPU_CUDA
      cuda_tune_wrapper_reduce<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cuda_impl",  IMPL_CUDA,   &cuda_tune_wrapper_reduce<StructType, StructType>));
#endif

      std::ofstream outfile(std::string("tree_data_" + id + ".dat").c_str());
      assert(outfile.good());

      extra.actDimensions = actDimens;
      Trainer trainer(impls, lowerBounds, upperBounds, MAX_DEPTH, nImpls, extra, callBackFunction, callBackFunctionMapReduce, OVERSAMPLE);
      trainer.train();

      ExecPlanNew<1> plan;
      trainer.constructExecPlanNew(&plan, stats);
      assert(plan.calibrated);
      outfile << plan ;

      outfile << "compressed plan:\n";
      trainer.compressExecPlanNew(plan);
      for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = plan.m_data.begin(); it != plan.m_data.end(); ++it)
      {
         switch(it->second)
         {
         case IMPL_CPU:
            bp.backend = CPU_BACKEND;
            break;
         case IMPL_OMP:
            bp.backend = OMP_BACKEND;
            break;
         case IMPL_CUDA:
            bp.backend = CU_BACKEND;
            break;
         default:
            assert(false);
         }
         execPlan.add(it->first.first, it->first.second, bp);
      }
      outfile << plan;

      /*! This functionality only for non-windows platform */
#ifndef _WIN32
      storeExecPlan(id, execPlan);
#endif

      outfile << *(trainer.m_tree);
      DEBUG_TUNING_LEVEL2( "\nTree: " << *(trainer.m_tree) << "\n");
      
      // free memory...
      for(int i=0; i<impls.size(); ++i)
      {
         delete impls[i];
      }

      impls.clear();

      return execPlan;
   }

public:
   void (*callBackFunction)(void*, size_t*, unsigned int);
   void (*callBackFunctionMapReduce)(void*, void*, size_t*, unsigned int);

private:
   ExtraData extra;
   unsigned int dimens;
   size_t *lowBounds;
   size_t *uppBounds;
   std::string id;
};


/*!
 *  \ingroup tuning
 */
/*!
 * \brief Tuner class specilization for MapArray skeleton
 */
template <typename StructType>
struct Tuner<StructType, MAPARRAY, StructType>
{
   Tuner(std::string _id, int _dimens, size_t *_lowBounds, size_t *_uppBounds): id(_id), dimens(_dimens), lowBounds(_lowBounds), uppBounds(_uppBounds), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(dimens == 3 && lowBounds && uppBounds);
      extra.memUp = NULL;
      extra.memDown = NULL;
   }

   Tuner(std::string _id, int _dimens, size_t *_lowBounds, size_t *_uppBounds, int *_memUp, int *_memDown): id(_id), dimens(_dimens), lowBounds(_lowBounds), uppBounds(_uppBounds), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(dimens == 3 && lowBounds && uppBounds);
      extra.memUp = _memUp;
      extra.memDown = _memDown;
   }

   StatsTuner stats;

   void operator()(ExecPlan *execPlanArray)
   {
      assert(execPlanArray!=NULL);
      
      /*!
       * Internal logic uses memup and memdown variable to find out whether to use hints or the second mode
       * memup and memdown should be null to make sure that they are not accessed
       */
      int *oldMemUp = extra.memUp;
      extra.memUp = NULL;
      int *oldMemDown = extra.memDown;
      extra.memDown = NULL;

      unsigned int actDimens = dimens;
      std::string interface = "maparray";

      bool allSame = ((lowBounds[0] == lowBounds[1]) && (lowBounds[1] == lowBounds[2])) && ((uppBounds[0] == uppBounds[1]) && (uppBounds[1] == uppBounds[2]));

      dimens = (allSame)? 1:2;

      /*! TODO: for simplification, fix later... */
      unsigned int nImpls = 1;
      if(!allSame)
            SKEPU_ERROR("The current tuning framework does not support MapArray skeleton tuning with different vector sizes. TODO in future.");
   #ifdef SKEPU_CUDA
         nImpls = 6;
   #endif
      assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);

      /*! This functionality only for non-windows platform */
#if !defined(_WIN32) && !defined(REDO_MEASUREMENTS)
      if(loadExecPlanArray(id, execPlanArray))
      {
         bool redoMesures = false;
         for(unsigned int i=0; i<nImpls; ++i)
         {
            if(execPlanArray[i].calibrated == false)
            {
               redoMesures = true;
               break;
            }
            for(unsigned int i=0; i<dimens; ++i)
            {
               if(execPlanArray[i].isTrainedFor(lowBounds[i]) == false || execPlanArray[i].isTrainedFor(uppBounds[i]) == false)
               {
                  redoMesures = true;
                  break;
               }
            }
         }
         /*! only use existing stored ExecPlan when it has same no of Impls and also trained (atleast) for the required input range  */
         if(redoMesures == false)
            return;
      }
#endif

      BackEndParams bp;
      createDefaultConfiguration(bp);

      /*!
       * TODO: For execPlan, we support only 1 dimension. Can be fixed in future to work with two dimensions...
       */
      assert(dimens == 1);

      std::vector<size_t> upperBounds(dimens);
      std::vector<size_t> lowerBounds(dimens);

      for(unsigned int i=0; i<dimens; ++i)
      {
         upperBounds[i] = uppBounds[i];
         lowerBounds[i] = lowBounds[i];
      }

      std::vector<ImpDetail*> impls;

      cpu_tune_wrapper_maparray<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cpu_impl",  IMPL_CPU,   &cpu_tune_wrapper_maparray<StructType, StructType>));

#ifdef SKEPU_OPENMP
      omp_tune_wrapper_maparray<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("omp_impl",  IMPL_OMP,   &omp_tune_wrapper_maparray<StructType, StructType>));
#endif

#ifdef SKEPU_CUDA
      cuda_tune_wrapper_maparray<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cuda_impl",  IMPL_CUDA,   &cuda_tune_wrapper_maparray<StructType, StructType>));
#endif

      std::ofstream outfile(std::string("tree_data_multi_" + id + ".dat").c_str());
      assert(outfile.good());

      extra.actDimensions = actDimens;
      Trainer trainer(impls, lowerBounds, upperBounds, MAX_DEPTH, nImpls, extra, callBackFunction, callBackFunctionMapReduce, OVERSAMPLE);
      trainer.train();

      ExecPlanNew<1> planArr[MAX_EXEC_PLANS];
      trainer.constructExecPlanNew(planArr, stats);

      for(unsigned int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         if(planArr[i].calibrated == false)
            break;

         outfile << planArr[i];
      }

      for(unsigned int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         if(planArr[i].calibrated == false)
            break;

         execPlanArray[i].clear();
         outfile << "compressed plan:\n";
         trainer.compressExecPlanNew(planArr[i]);
         for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = planArr[i].m_data.begin(); it != planArr[i].m_data.end(); ++it)
         {
            switch(it->second)
            {
            case IMPL_CPU:
               bp.backend = CPU_BACKEND;
               break;
            case IMPL_OMP:
               bp.backend = OMP_BACKEND;
               break;
            case IMPL_CUDA:
               bp.backend = CU_BACKEND;
               break;
            default:
               assert(false);
            }
            execPlanArray[i].add(it->first.first, it->first.second, bp);
            execPlanArray[i].calibrated = true;
         }
         outfile << planArr[i];
      }

      /*! This functionality only for non-windows platform */
#ifndef _WIN32
      storeExecPlanArray(id, execPlanArray, nImpls);
#endif

      outfile << *(trainer.m_tree);
      DEBUG_TUNING_LEVEL2( "\nTree: " << *(trainer.m_tree) << "\n");

      // free memory...
      for(unsigned int i=0; i<impls.size(); ++i)
      {
         delete impls[i];
      }

      impls.clear();
      
      // restore them now...
      extra.memUp = oldMemUp;
      extra.memDown = oldMemDown;
   }
   
   
   ExecPlan operator()()
   {
      assert(extra.memUp != NULL && extra.memDown != NULL);

      unsigned int actDimens = dimens;
      std::string interface = "maparray";
      
      bool allSame = ((lowBounds[0] == lowBounds[1]) && (lowBounds[1] == lowBounds[2])) && ((uppBounds[0] == uppBounds[1]) && (uppBounds[1] == uppBounds[2]));

      dimens = (allSame)? 1:2;

      unsigned int nImpls = 1;
      if(!allSame)
            SKEPU_ERROR("The current tuning framework does not support MapArray skeleton tuning with different vector sizes. TODO in future.");
      
      ExecPlan execPlan;

      /*! This functionality only for non-windows platform */
#if !defined(_WIN32) && !defined(REDO_MEASUREMENTS)
      if(loadExecPlan(id, execPlan))
      {
         return execPlan;
      }
#endif

      BackEndParams bp;
      createDefaultConfiguration(bp);

      std::vector<size_t> upperBounds(dimens);
      std::vector<size_t> lowerBounds(dimens);

      for(unsigned int i=0; i<dimens; ++i)
      {
         upperBounds[i] = uppBounds[i];
         lowerBounds[i] = lowBounds[i];
      }

      std::vector<ImpDetail*> impls;

      cpu_tune_wrapper_maparray<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cpu_impl",  IMPL_CPU,   &cpu_tune_wrapper_maparray<StructType, StructType>));

#ifdef SKEPU_OPENMP
      omp_tune_wrapper_maparray<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("omp_impl",  IMPL_OMP,   &omp_tune_wrapper_maparray<StructType, StructType>));
#endif

#ifdef SKEPU_CUDA
      cuda_tune_wrapper_maparray<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cuda_impl",  IMPL_CUDA,   &cuda_tune_wrapper_maparray<StructType, StructType>));
#endif

      std::ofstream outfile(std::string("tree_data_" + id + ".dat").c_str());
      assert(outfile.good());

      extra.actDimensions = actDimens;
      Trainer trainer(impls, lowerBounds, upperBounds, MAX_DEPTH, nImpls, extra, callBackFunction, callBackFunctionMapReduce, OVERSAMPLE);
      trainer.train();

      ExecPlanNew<1> plan;
      trainer.constructExecPlanNew(&plan, stats);
      assert(plan.calibrated);
      outfile << plan ;

      outfile << "compressed plan:\n";
      trainer.compressExecPlanNew(plan);
      for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = plan.m_data.begin(); it != plan.m_data.end(); ++it)
      {
         switch(it->second)
         {
         case IMPL_CPU:
            bp.backend = CPU_BACKEND;
            break;
         case IMPL_OMP:
            bp.backend = OMP_BACKEND;
            break;
         case IMPL_CUDA:
            bp.backend = CU_BACKEND;
            break;
         default:
            assert(false);
         }
         execPlan.add(it->first.first, it->first.second, bp);
      }
      outfile << plan;

      /*! This functionality only for non-windows platform */
#ifndef _WIN32
      storeExecPlan(id, execPlan);
#endif

      outfile << *(trainer.m_tree);
      DEBUG_TUNING_LEVEL2( "\nTree: " << *(trainer.m_tree) << "\n");
      
      // free memory...
      for(int i=0; i<impls.size(); ++i)
      {
         delete impls[i];
      }

      impls.clear();

      return execPlan;
   }

public:
   void (*callBackFunction)(void*, size_t*, unsigned int);
   void (*callBackFunctionMapReduce)(void*, void*, size_t*, unsigned int);

private:
   ExtraData extra;
   unsigned int dimens;
   size_t *lowBounds;
   size_t *uppBounds;
   std::string id;
};



/*!
 *  \ingroup tuning
 */
/*!
 * \brief Tuner class specilization for MapOverlap skeleton
 */
template <typename StructType>
struct Tuner<StructType, MAPOVERLAP, StructType>
{
   Tuner(std::string _id, int _dimens, size_t *_lowBounds, size_t *_uppBounds): id(_id), dimens(_dimens), lowBounds(_lowBounds), uppBounds(_uppBounds), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(dimens >= 1 && dimens <= 2 && lowBounds && uppBounds);
      extra.memUp = NULL;
      extra.memDown = NULL;
   }

   Tuner(std::string _id, int _dimens, size_t *_lowBounds, size_t *_uppBounds, int *_memUp, int *_memDown): id(_id), dimens(_dimens), lowBounds(_lowBounds), uppBounds(_uppBounds), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(dimens >= 1 && dimens <= 2 && lowBounds && uppBounds);
      extra.memUp = _memUp;
      extra.memDown = _memDown;
   }
   
   StatsTuner stats;
   
   void operator()(ExecPlan *execPlanArray)
   {
      assert(execPlanArray!=NULL);
      
      /*!
       * Internal logic uses memup and memdown variable to find out whether to use hints or the second mode
       * memup and memdown should be null to make sure that they are not accessed
       */
      int *oldMemUp = extra.memUp;
      extra.memUp = NULL;
      int *oldMemDown = extra.memDown;
      extra.memDown = NULL;
      
      unsigned int actDimens = dimens;
      std::string interface = "mapoverlap";
      dimens = 1; /*! TODO check if it is always the case... */

      unsigned int nImpls = 1;
   #ifdef SKEPU_CUDA
         nImpls = 3;
   #endif
      assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);

      /*! This functionality only for non-windows platform */
#if !defined(_WIN32) && !defined(REDO_MEASUREMENTS)
      if(loadExecPlanArray(id, execPlanArray))
      {
         bool redoMesures = false;
         for(unsigned int i=0; i<nImpls; ++i)
         {
            if(execPlanArray[i].calibrated == false)
            {
               redoMesures = true;
               break;
            }
            for(int j=0; j<dimens; ++j)
            {
               if(execPlanArray[i].isTrainedFor(lowBounds[j]) == false || execPlanArray[i].isTrainedFor(uppBounds[j]) == false)
               {
                  redoMesures = true;
                  break;
               }
            }
         }
         /*! only use existing stored ExecPlan when it has same no of Impls and also trained (atleast) for the required input range  */
         if(redoMesures == false)
            return;
      }
#endif

      BackEndParams bp;
      createDefaultConfiguration(bp);

      std::vector<size_t> upperBounds(dimens);
      std::vector<size_t> lowerBounds(dimens);

      for(unsigned int i=0; i<dimens; ++i)
      {
         upperBounds[i] = uppBounds[i];
         lowerBounds[i] = lowBounds[i];
      }

      std::vector<ImpDetail*> impls;

      cpu_tune_wrapper_mapoverlap<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cpu_impl",  IMPL_CPU,   &cpu_tune_wrapper_mapoverlap<StructType, StructType>));

#ifdef SKEPU_OPENMP
      omp_tune_wrapper_mapoverlap<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("omp_impl",  IMPL_OMP,   &omp_tune_wrapper_mapoverlap<StructType, StructType>));
#endif

#ifdef SKEPU_CUDA
      cuda_tune_wrapper_mapoverlap<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cuda_impl",  IMPL_CUDA,   &cuda_tune_wrapper_mapoverlap<StructType, StructType>));
#endif

      std::ofstream outfile(std::string("tree_data_multi_" + id + ".dat").c_str());
      assert(outfile.good());

      extra.actDimensions = actDimens;
      Trainer trainer(impls, lowerBounds, upperBounds, MAX_DEPTH, nImpls, extra, callBackFunction, callBackFunctionMapReduce, OVERSAMPLE);
      trainer.train();


      ExecPlanNew<1> planArr[MAX_EXEC_PLANS];
      trainer.constructExecPlanNew(planArr, stats);

      for(unsigned int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         if(planArr[i].calibrated == false)
            break;

         outfile << planArr[i];
      }

      for(unsigned int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         if(planArr[i].calibrated == false)
            break;

         execPlanArray[i].clear();
         outfile << "compressed plan:\n";
         trainer.compressExecPlanNew(planArr[i]);
         for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = planArr[i].m_data.begin(); it != planArr[i].m_data.end(); ++it)
         {
            switch(it->second)
            {
            case IMPL_CPU:
               bp.backend = CPU_BACKEND;
               break;
            case IMPL_OMP:
               bp.backend = OMP_BACKEND;
               break;
            case IMPL_CUDA:
               bp.backend = CU_BACKEND;
               break;
            default:
               assert(false);
            }
            execPlanArray[i].add(it->first.first, it->first.second, bp);
            execPlanArray[i].calibrated = true;
         }
         outfile << planArr[i];
      }

      /*! This functionality only for non-windows platform */
#ifndef _WIN32
      storeExecPlanArray(id, execPlanArray, nImpls);
#endif

      outfile << *(trainer.m_tree);
      DEBUG_TUNING_LEVEL2( "\nTree: " << *(trainer.m_tree) << "\n");

      // free memory...
      for(unsigned int i=0; i<impls.size(); ++i)
      {
         delete impls[i];
      }

      impls.clear();
      
      // restore them now...
      extra.memUp = oldMemUp;
      extra.memDown = oldMemDown;
   }
   
   
   ExecPlan operator()()
   {
      assert(extra.memUp != NULL && extra.memDown != NULL);

      unsigned int actDimens = dimens;
      std::string interface = "mapoverlap";
      
      dimens = 1;

      unsigned int nImpls = 1;
      
      ExecPlan execPlan;
      
      /*! This functionality only for non-windows platform */
#if !defined(_WIN32) && !defined(REDO_MEASUREMENTS)
      if(loadExecPlan(id, execPlan))
      {
         return execPlan;
      }
#endif

      BackEndParams bp;
      createDefaultConfiguration(bp);

      std::vector<size_t> upperBounds(dimens);
      std::vector<size_t> lowerBounds(dimens);

      for(unsigned int i=0; i<dimens; ++i)
      {
         upperBounds[i] = uppBounds[i];
         lowerBounds[i] = lowBounds[i];
      }

      std::vector<ImpDetail*> impls;

      cpu_tune_wrapper_mapoverlap<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cpu_impl",  IMPL_CPU,   &cpu_tune_wrapper_mapoverlap<StructType, StructType>));

#ifdef SKEPU_OPENMP
      omp_tune_wrapper_mapoverlap<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("omp_impl",  IMPL_OMP,   &omp_tune_wrapper_mapoverlap<StructType, StructType>));
#endif

#ifdef SKEPU_CUDA
      cuda_tune_wrapper_mapoverlap<StructType, StructType>(0); /*! dummy call to pass by nvcc compiler error */
      impls.push_back(new ImpDetail("cuda_impl",  IMPL_CUDA,   &cuda_tune_wrapper_mapoverlap<StructType, StructType>));
#endif

      std::ofstream outfile(std::string("tree_data_" + id + ".dat").c_str());
      assert(outfile.good());

      extra.actDimensions = actDimens;
      Trainer trainer(impls, lowerBounds, upperBounds, MAX_DEPTH, nImpls, extra, callBackFunction, callBackFunctionMapReduce, OVERSAMPLE);
      trainer.train();

      ExecPlanNew<1> plan;
      trainer.constructExecPlanNew(&plan, stats);
      assert(plan.calibrated);
      outfile << plan ;

      outfile << "compressed plan:\n";
      trainer.compressExecPlanNew(plan);
      for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = plan.m_data.begin(); it != plan.m_data.end(); ++it)
      {
         switch(it->second)
         {
         case IMPL_CPU:
            bp.backend = CPU_BACKEND;
            break;
         case IMPL_OMP:
            bp.backend = OMP_BACKEND;
            break;
         case IMPL_CUDA:
            bp.backend = CU_BACKEND;
            break;
         default:
            assert(false);
         }
         execPlan.add(it->first.first, it->first.second, bp);
      }
      outfile << plan;

      /*! This functionality only for non-windows platform */
#ifndef _WIN32
      storeExecPlan(id, execPlan);
#endif

      outfile << *(trainer.m_tree);
      DEBUG_TUNING_LEVEL2( "\nTree: " << *(trainer.m_tree) << "\n");
      
      // free memory...
      for(int i=0; i<impls.size(); ++i)
      {
         delete impls[i];
      }

      impls.clear();

      return execPlan;
   }

public:
   void (*callBackFunction)(void*, size_t*, unsigned int);
   void (*callBackFunctionMapReduce)(void*, void*, size_t*, unsigned int);

private:
   ExtraData extra;
   unsigned int dimens;
   size_t *lowBounds;
   size_t *uppBounds;
   std::string id;
};



} // end namespace skepu....



#endif