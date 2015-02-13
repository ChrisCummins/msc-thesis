#ifndef __TRAINER__H_
#define __TRAINER__H_


#include <iostream>
#include <cassert>
#include <map>
#include "environment.h"


#ifndef TRAINING_RUNS
#define TRAINING_RUNS 1
#endif

namespace skepu
{


/*!
 *  \ingroup tuning
 */
template<int T>
struct identity
{
   const static int data = T;
};


/*!
 *  \ingroup tuning
 */
template <unsigned int dimens>
struct ExecPlanNew
{
};

/*!
 *  \ingroup tuning
 */
template <>
struct ExecPlanNew<1>
{
   std::map<std::pair<size_t,size_t>, ImplType> m_data;
   bool calibrated;
   int idx;

   ExecPlanNew<1>() : calibrated(false), idx(-1) {}
};

/*!
 *  \ingroup tuning
 */
struct Point2D
{
   size_t dim1;
   size_t dim2;

   Point2D() {}

   Point2D(const Point2D& copy)
   {
      dim1 = copy.dim1;
      dim2 = copy.dim2;
   }
   Point2D(size_t _d1, size_t _d2): dim1(_d1), dim2(_d2)
   {  }

   bool operator < (const Point2D & p) const
   {
      if (dim1 == p.dim1)
         return (dim2 < p.dim2);

      return (dim1 < p.dim1);
   }
};

/*!
 *  \ingroup tuning
 */
template <>
struct ExecPlanNew<2>
{
   std::map< std::pair<Point2D, Point2D>, ImplType> m_data;
   bool isOpen;
   std::multimap< std::pair<Point2D, Point2D>, std::pair<Point2D, ImplType> > openPoints;

   ExecPlanNew<2>() : isOpen(false), calibrated(false), idx(-1) {}

   bool calibrated;
   int idx;
};

int treeNodeId = 0;

/*!
 *  \ingroup tuning
 *  \brief Any extra information that User want to pass to the function wrapper for implementations can be specified here....
 */ 
struct ExtraData
{
   ExtraData(): actDimensions(-1), memUp(0), memDown(0)
   {}
   int actDimensions;
   int *memUp;
   int *memDown;
};


/*!
 *  \ingroup tuning
 */
struct ImpDetail
{
   ImpDetail(std::string _impName, ImplType _impType, void (*_impPtr)(void*)):impName(_impName), impType(_impType), impPtr(_impPtr)
   {}

   std::string impName;
   ImplType impType;

   void (*impPtr)(void*);
};

/*!
 *  \ingroup tuning
 */
struct TrainingData
{
public:
   TrainingData(std::vector<size_t> &_problemSize, unsigned int _nImpls): problemSize(_problemSize), nImpls(_nImpls), extra(NULL), callBackFunction(NULL), callBackFunctionMapReduce(NULL)
   {
      assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
      for(size_t i=0; i<nImpls; ++i)
      {
         exec_time[i] = -1;
      }
   }
   unsigned int dimens;

   /*! localityIdx determines where operand data is.. its value is from 0...nImpls.
    *  0 means no operand is in GPU memory...
    *  1 means 1 operand in in GPU memory...
    *  2 means 2 operands are in GPU memory and so on...
    *  TODO: very simple, works assuming all operand data are of same size and assumes 1 main memory and 1 gpu memory model. i.e. does not work for multi-gpu case...
    */
//     int localityIdx;

   unsigned int nImpls;

   ExtraData *extra;
   std::vector<size_t> &problemSize;
   double exec_time[MAX_EXEC_PLANS];
   void (*callBackFunction)(void*, size_t*, unsigned int);
   void (*callBackFunctionMapReduce)(void*, void*, size_t*, unsigned int);
};


/*!
 *  \ingroup tuning
 */
/*!
 * A point represent one combination of input size(s) that is tried....
 * Now we changes it to represent multiple possible execution configurations depending upon
 * the oeprand data locality...
 * It represent information about each implementation type and best performing among them...
 */
struct Point
{
public:
   Point(std::vector<size_t> _problemSize, unsigned int _nImpls) : problemSize(_problemSize), nImpls(_nImpls)
   {
      assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
      for(unsigned int i=0; i<nImpls; ++i)
      {
         cpuImpTime[i].second  = -1;
         ompImpTime[i].second  = -1;
         cudaImpTime[i].second = -1;
         bestImpTime[i].second = -1;
         cpuImpTime[i].first   = NULL;
         ompImpTime[i].first   = NULL;
         cudaImpTime[i].first  = NULL;
         bestImpTime[i].first  = NULL;
      }
   }
   std::vector<size_t> problemSize;
   std::pair<ImpDetail*, double> cpuImpTime[MAX_EXEC_PLANS];
   std::pair<ImpDetail*, double> ompImpTime[MAX_EXEC_PLANS];
   std::pair<ImpDetail*, double> cudaImpTime[MAX_EXEC_PLANS];
   std::pair<ImpDetail*, double> bestImpTime[MAX_EXEC_PLANS];

   /*! actual implementations that are available */
   unsigned int nImpls;
};


/*!
 *  \ingroup helper
 */
std::string getStringImplType(ImplType type)
{
   switch(type)
   {
   case IMPL_CPU:
      return "CPU";
   case IMPL_OMP:
      return "OMP";
   case IMPL_CUDA:
      return "CUDA";
   default:
      assert(false);
   }
   return "";
}

/*!
 *  \ingroup tuning
 */
struct StatsTuner
{
   unsigned int maxDepth;
   unsigned int numLeafClosedRanges;
   unsigned int numLeafTotalRanges;
   unsigned int numTotalRanges;
   unsigned int numTrainingPoints; // marks total number of points that are traversed...
   unsigned int numTrainingExec;
};


/*!
 *  \ingroup tuning
 */
/*!
 * A node represents one node of a tree which can have a parent node and zero or more child nodes...
 * Each node can have two or more points on which it is measured... For dimens = 1, each node will have 2 points and so on...
 */
struct Node
{
public:
   Node(std::vector<size_t> &_lowBounds, std::vector<size_t> &_uppBounds, unsigned int _nImpls): lowBounds(_lowBounds), uppBounds(_uppBounds), level(0), father(NULL), id(-1), nImpls(_nImpls)
   {
      for(int i=0; i<MAX_EXEC_PLANS; ++i)
      {
         bestNodeImp[i] = NULL;
         isClosed[i] = false;
      }
   }

   unsigned int nImpls;
   std::vector<Point*> points; // represent each border point....
   std::vector<size_t> uppBounds; // represent low bounds of this area covered by this node...
   std::vector<size_t> lowBounds; // represent upper bounds of this area covered by this node...
   Node *father;
   int id;
   std::vector<Node*> childs;
   ImpDetail* bestNodeImp[MAX_EXEC_PLANS];
   bool isClosed[MAX_EXEC_PLANS];
   unsigned int level;

   /*!
    * A friend function that can print a Node object
    */
   friend std::ostream& operator<<(std::ostream& os, const Node& node)
   {
      std::string padding = "";
      for(unsigned int i=0; i<node.level; ++i)
         padding += "----";
      os << padding << "\"" << node.level << "\" ( ";
      for (unsigned int i=0; i<node.lowBounds.size(); ++i)
      {
         if(i!= node.lowBounds.size()-1)
            os << node.lowBounds[i] << ",";
         else
            os << node.lowBounds[i];
      }
      os << " --- ";
      for (unsigned int i=0; i<node.uppBounds.size(); ++i)
      {
         if(i!= node.uppBounds.size()-1)
            os << node.uppBounds[i] << ",";
         else
            os << node.uppBounds[i];
      }
      for(unsigned int i=0; i<node.nImpls; ++i)
      {
         if(i == 0)
            os << ") ";

         if(node.isClosed[i])
         {
            assert(node.bestNodeImp[i] != NULL);
            os << " " << getStringImplType(node.bestNodeImp[i]->impType) << " ";
         }
         else
            os << " [OPEN] ";

         if(i == (node.nImpls-1))
            os << "\n";
      }

      for(unsigned int i=0; i<node.childs.size(); ++i)
      {
         os << *(node.childs[i]);
      }
      return os;
   }


   /*!
    * Node class destructor that recursively deletes all node child objects
    */
   ~Node() // need to just call delete on root node as it internally delete recursively....
   {
      if(childs.empty() == false) // non-leaf node...
      {
         // first delete child nodes, recursively...
         for(unsigned int i=0; i<childs.size(); ++i)
         {
//             DEBUG_TUNING_LEVEL3("Recursive *delete["<<i<<"]->points.size(): " << childs[i]->points.size() << "\n");
            delete childs[i];
         }
         childs.clear();

         for(unsigned int i=0; i<points.size(); ++i)
         {
            delete points[i];
         }
         points.clear();
      }
      else // leaf node...
      {
         for(unsigned int i=0; i<points.size(); ++i)
         {
            delete points[i];
         }
         points.clear();
      }
   }


   template <int dimens>
   void constructExecPlanNew(ExecPlanNew<dimens> &plan, StatsTuner &stats, int idx)
   {
      constructExecPlanNew(plan, stats, idx, identity<dimens>());
   }



private:

   template <unsigned int dimens>
   void constructExecPlanNew(ExecPlanNew<dimens> &plan, StatsTuner &stats, int idx, identity<dimens>)
   {
      assert(false);
   }

   void constructExecPlanNew(ExecPlanNew<1> &plan, StatsTuner &stats, int idx, identity<1>)
   {
      assert(uppBounds.size() == lowBounds.size() && uppBounds.size() == 1);

      stats.numTrainingPoints += 2;
      stats.numTotalRanges++;

      if(childs.empty() == false)
      {
         for(unsigned int i=0; i<childs.size(); ++i)
         {
//             DEBUG_TUNING_LEVEL2("Recursive *childs["<<i<<"]->points.size(): " << childs[i]->points.size() << "\n");
            childs[i]->constructExecPlanNew<1>(plan, stats, idx);
         }
      }
      else // leaf nodes...
      {
         stats.numLeafTotalRanges++;

         if(isClosed[idx])
         {
            stats.numLeafClosedRanges++;

            DEBUG_TUNING_LEVEL3("Closed space: " << points.size() << ",  " << lowBounds[0] << " - " << uppBounds[0] << "\n");
            assert(bestNodeImp[idx] != NULL);
            assert( plan.m_data.insert(std::make_pair(std::make_pair(lowBounds[0], uppBounds[0]), bestNodeImp[idx]->impType)).second == true);
         }
         else
         {
            assert(points.size() > 1);
            DEBUG_TUNING_LEVEL3("Open space: " << points.size() << ",  " << lowBounds[0] << " - " << uppBounds[0] << "\n");
            
            /*! segmentation fault.... need to correct it.... */
            assert(points[0]->bestImpTime[idx].first != NULL && points[1]->bestImpTime[idx].first != NULL);
            assert(points[0]->bestImpTime[idx].first->impType != points[1]->bestImpTime[idx].first->impType);

            std::pair<ImplType, double> bestPoint1(points[0]->bestImpTime[idx].first->impType, points[0]->bestImpTime[idx].second);
            std::pair<ImplType, double> bestPoint2(points[1]->bestImpTime[idx].first->impType, points[1]->bestImpTime[idx].second);

            double secBestPoint1 = 0;
            switch(points[1]->bestImpTime[idx].first->impType)
            {
            case IMPL_CPU:
               secBestPoint1 = points[0]->cpuImpTime[idx].second;
               break;
            case IMPL_OMP:
               secBestPoint1 = points[0]->ompImpTime[idx].second;
               break;
            case IMPL_CUDA:
               secBestPoint1 = points[0]->cudaImpTime[idx].second;
               break;
            default:
               assert(false);
            }

            std::pair<ImplType, double> secondBestPoint1(points[1]->bestImpTime[idx].first->impType, secBestPoint1);

            double secBestPoint2 = 0;
            switch(points[0]->bestImpTime[idx].first->impType)
            {
            case IMPL_CPU:
               secBestPoint2 = points[1]->cpuImpTime[idx].second;
               break;
            case IMPL_OMP:
               secBestPoint2 = points[1]->ompImpTime[idx].second;
               break;
            case IMPL_CUDA:
               secBestPoint2 = points[1]->cudaImpTime[idx].second;
               break;
            default:
               assert(false);
            }

            std::pair<ImplType, double> secondBestPoint2(points[0]->bestImpTime[idx].first->impType, secBestPoint2);

// 		std::cerr << "Point1: Best:   " << getStringImplType(bestPoint1.first) << ", " << bestPoint1.second << "\n";
// 		std::cerr << "Point2: Best:   " << getStringImplType(bestPoint2.first) << ", " << bestPoint2.second << "\n";
// 		std::cerr << "Point1: Second: " << getStringImplType(secondBestPoint1.first) << ", " << secondBestPoint1.second << "\n";
// 		std::cerr << "Point2: Second: " << getStringImplType(secondBestPoint2.first) << ", " << secondBestPoint2.second << "\n";

            //Line1
            double A1 = secondBestPoint2.second - bestPoint1.second; //sorted[lastBestImpl][i].second.first - sorted[lastBestImpl][lastIdx].second.first;
            ssize_t B1 = lowBounds[0] - uppBounds[0]; // sorted[lastBestImpl][lastIdx].first - sorted[lastBestImpl][i].first;
            double C1 = A1*(lowBounds[0]) + B1*(bestPoint1.second);

            double A2 = bestPoint2.second - secondBestPoint1.second; //sorted[lastBestImpl][i].second.first - sorted[lastBestImpl][lastIdx].second.first;
            ssize_t B2 = B1; // sorted[lastBestImpl][lastIdx].first - sorted[lastBestImpl][i].first;
            double C2 = A2*(lowBounds[0]) + B2*(secondBestPoint1.second);

            double delta = A1*B2 - A2*B1;

            assert(delta != 0);

            size_t x = (size_t)((B2*C1 - B1*C2)/delta);

            DEBUG_TUNING_LEVEL3("------------------------------------------\n");
            DEBUG_TUNING_LEVEL3("(" << lowBounds[0] << "-" << x << ") " << getStringImplType(bestPoint1.first) << "\n");
            DEBUG_TUNING_LEVEL3("(" << (x+1) << "-" << uppBounds[0] << ") " << getStringImplType(bestPoint2.first) << "\n");
            DEBUG_TUNING_LEVEL3("------------------------------------------\n");

            assert( plan.m_data.insert(std::make_pair(std::make_pair(lowBounds[0], x), bestPoint1.first)).second   == true );
            assert( plan.m_data.insert(std::make_pair(std::make_pair(x+1, uppBounds[0]), bestPoint2.first)).second == true ); // to ensure that data is inserted always...
         }
      }

      if(plan.m_data.empty() == false)
         plan.calibrated = true;

      plan.idx = idx;
   }
   
}; /// end Node class...





/*!
 *  \ingroup tuning
 */
/*!
 * \brief Trainer class which is called by the Tuner class for managing the actual training process.
 */
class Trainer
{
public:
   Trainer(std::vector<ImpDetail*> &impls, std::vector<size_t> &lowerBounds, std::vector<size_t> &upperBounds, unsigned int maxDepth, unsigned int _nImpls, ExtraData &extra, void (*_callBack1)(void*, size_t*, unsigned int), void (*_callBack2)(void*, void*, size_t*, unsigned int), bool Oversampling = false);
   void train();

   template <unsigned int dimens>
   void constructExecPlanNew(ExecPlanNew<dimens> *plan, StatsTuner &stats);

   /*!
    * This method compresses an execution plan
    */
   void compressExecPlanNew(ExecPlanNew<1> &plan)
   {
      ExecPlanNew<1> newPlan;
      std::pair< std::pair<size_t, size_t >, ImplType> prevBest;
      bool init=false;
      for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = plan.m_data.begin(); it != plan.m_data.end(); ++it)
      {
         if(!init)
         {
            prevBest = (*it);
            init = true;
            continue;
         }

         if(it->second == prevBest.second)
         {
            assert(prevBest.first.second = it->first.first-1);
            prevBest.first.second = it->first.second;
         }
         else
         {
            assert(newPlan.m_data.insert(prevBest).second == true);
            prevBest = (*it);
         }
      }

      if(init) // add the last one after the loop...
         assert(newPlan.m_data.insert(prevBest).second == true);

      newPlan.idx = plan.idx;
      plan = newPlan;
   }

   ~Trainer()
   {
      if(m_tree != NULL)
         delete m_tree;

      m_tree = NULL;
   }



private:
   void train(Node* &rootNode, bool discardFirst);
   void generateSubSpaces(std::vector<size_t> &baseLowBound, std::vector<size_t> &baseUppBound, Node *father);
   void generateAllPossibleCombinations(std::vector<size_t> &lowBound, std::vector<size_t> &uppBound, std::vector<std::vector<size_t> > &combinations, bool overSample);

   std::vector<size_t> &m_lowerBounds;
   std::vector<size_t> &m_upperBounds;
   unsigned int m_maxDepth;
   bool m_overSampling;
   ExtraData &m_extra;
   std::vector<ImpDetail*> &m_impls;
   void (*callBackFunction)(void*, size_t*, unsigned int);
   void (*callBackFunctionMapReduce)(void*, void*, size_t*, unsigned int);

   unsigned int nImpls;

public:
   Node *m_tree; //[MAX_EXEC_PLANS];
};

} // end namespace skepu

#include "trainer.inl"

#endif

