#include <iostream>
#include <cassert>
#include <cmath>



namespace skepu
{

/*!
 *  \ingroup tuning
 */
/*!
 * The helper method that can print the ExecPlanNew<1> objects
 */   
std::ostream& operator<<(std::ostream &os, ExecPlanNew<1> &plan)
{
   os << "--------------------------------------\n";
   os << "# " << plan.idx << "\n";
//    os << "--------------------------------------\n";
   for(std::map<std::pair<size_t,size_t>, ImplType>::iterator it = plan.m_data.begin(); it != plan.m_data.end(); it++)
   {
      os << it->first.first << " - " << it->first.second << " ==> " << getStringImplType(it->second) << "\n";
   }
   os << "--------------------------------------\n";
   return os;
}


/*!
 *  \ingroup tuning
 */
/*!
 * The helper method that can print the ExecPlanNew<2> objects
 */  
std::ostream& operator<<(std::ostream &os, ExecPlanNew<2> &plan)
{
   os << "--------------------------------------------\n";
   os << "# " << plan.idx << "\n";
   os << "--------------------------------------------\n";
   for(std::map<std::pair<Point2D,Point2D>, ImplType>::iterator it = plan.m_data.begin(); it != plan.m_data.end(); it++)
   {
      if(plan.isOpen)
      {
         typedef std::multimap< std::pair<Point2D, Point2D>, std::pair<Point2D, ImplType> > mm;
         std::multimap< std::pair<Point2D, Point2D>, std::pair<Point2D, ImplType> > openPoints;
         assert(plan.openPoints.find(std::make_pair(it->first.first, it->first.second)) != plan.openPoints.end());

         std::pair <mm::iterator, mm::iterator> ret;
         ret = plan.openPoints.equal_range(std::make_pair(it->first.first, it->first.second));
         os << it->first.first.dim1 << "," << it->first.first.dim2 << " - " << it->first.second.dim1 << "," << it->first.second.dim2 << " ==> (";
         for (mm::iterator it2=ret.first; it2!=ret.second; ++it2)
            os << ' ' << getStringImplType(it2->second.second);

         os << ") [OPEN]\n";
      }
      else
         os << it->first.first.dim1 << "," << it->first.first.dim2 << " - " << it->first.second.dim1 << "," << it->first.second.dim2 << " ==> " << getStringImplType(it->second) << " [CLOSED]\n";
   }
   os << "--------------------------------------------\n";
   return os;
}



/*!
 * Trainer class constructor
 */  
Trainer::Trainer(std::vector<ImpDetail*> &impls, std::vector<size_t> &lowerBounds, std::vector<size_t> &upperBounds, unsigned int maxDepth, unsigned int _nImpls, ExtraData &extra, void (*_callBack1)(void*, size_t*, unsigned int), void (*_callBack2)(void*, void*, size_t*, unsigned int), bool overSampling): m_impls(impls), m_lowerBounds(lowerBounds), m_upperBounds(upperBounds), m_maxDepth(maxDepth), nImpls(_nImpls), m_extra(extra), m_overSampling(overSampling), m_tree(NULL), callBackFunction(_callBack1), callBackFunctionMapReduce(_callBack2)
{
   assert(m_impls.empty() == false);
}


/*!
 * Trainer class train method which is invoked by the Tuner class. It internally creates a node object and calls node train for actual working.
 */ 
void Trainer::train()
{
   m_tree = new Node(m_lowerBounds, m_upperBounds, nImpls);
   m_tree->level = 0;
   train(m_tree, true);
}


/*! To display vector contents... */
std::ostream &operator<<(std::ostream &os, const std::vector<size_t> &vec)
{
   for (size_t i=0; i<vec.size(); ++i)
      os << " " << vec[i];

   os << "\n";

   return os;
}



/*!
 * This method constructs a new execution plan
 */ 
template <unsigned int dimens>
void Trainer::constructExecPlanNew(ExecPlanNew<dimens> *plan, StatsTuner &stats)
{
//    bool first = true;
   StatsTuner statsTemp;
//    StatsTuner statsOld;
   for(unsigned int i=0; i<nImpls; ++i)
   {
      statsTemp.maxDepth = m_maxDepth;
      statsTemp.numLeafClosedRanges = 0;
      statsTemp.numLeafTotalRanges = 0;
      statsTemp.numTotalRanges = 0;
      statsTemp.numTrainingPoints = 0;
      statsTemp.numTrainingExec = 0;

//       plan[i].id = i;
      m_tree->constructExecPlanNew<dimens>(plan[i], statsTemp, i);
//       plan[i].calibrated = true;

//       if(!first)
//       {
//          assert(statsTemp.maxDepth            == statsOld.maxDepth);
//          assert(statsTemp.numLeafClosedRanges == statsOld.numLeafClosedRanges);
//          assert(statsTemp.numLeafTotalRanges  == statsOld.numLeafTotalRanges);
//          assert(statsTemp.numTotalRanges      == statsOld.numTotalRanges);
//          assert(statsTemp.numTrainingPoints   == statsOld.numTrainingPoints);
//          assert(statsTemp.numTrainingExec     == statsOld.numTrainingExec);
//       }
//       else
//          statsOld = statsTemp;

     
      DEBUG_TUNING_LEVEL2(plan[i]);
   }
   stats = statsTemp;

   stats.numTrainingExec = stats.numTrainingPoints * TRAINING_RUNS;
}

/*!
 * This method starts training from the root training node and build recursively the tree
 */
void Trainer::train(Node* &rootNode, bool discardFirst)
{
   if(m_maxDepth<1)
      m_maxDepth = 10;

   int depth = 0;

   std::vector<Node*> arrNodes;
   arrNodes.push_back(rootNode);

   unsigned int nImpls = rootNode->nImpls;
   assert(nImpls > 0 && nImpls <= MAX_EXEC_PLANS);
   while(true)
   {
      for(unsigned int n=0; n<arrNodes.size(); ++n)
      {
         std::vector<size_t> &lowBound = arrNodes[n]->lowBounds;
         std::vector<size_t> &uppBound = arrNodes[n]->uppBounds;

         DEBUG_TUNING_LEVEL2("****************************************\n");
         DEBUG_TUNING_LEVEL2("lowBound: " << lowBound);
         DEBUG_TUNING_LEVEL2("uppBound: " << uppBound);
         DEBUG_TUNING_LEVEL2("****************************************\n");

         std::vector< std::vector<size_t> > combinations;
         generateAllPossibleCombinations(lowBound, uppBound, combinations, m_overSampling);

         assert(combinations.size() >= 2);

         for(unsigned int i=0; i<combinations.size(); ++i)
         {
            assert(combinations[i].empty() == false);

            double bestPointTime[MAX_EXEC_PLANS];
            ImpDetail *bestPointImp[MAX_EXEC_PLANS];
            Point *point = new Point(combinations[i], nImpls);
            for(unsigned int k=0; k<nImpls; ++k)
            {
// 		    points[k] = new Point(combinations[i], nImpls);
               bestPointImp[k] = NULL;
               bestPointTime[k] = -1;
            }

            for(unsigned int impIdx = 0; impIdx<m_impls.size(); ++impIdx)
            {
               ImpDetail *imp = m_impls[impIdx];
               assert(imp != NULL);

               TrainingData arg (combinations[i], nImpls);
               arg.dimens = combinations[i].size();
               arg.extra = &m_extra;

               arg.callBackFunction = callBackFunction;
               arg.callBackFunctionMapReduce = callBackFunctionMapReduce;

               assert(imp->impPtr != NULL);

#ifdef DISCARD_FIRST_EXEC
               discardFirst = true;
#endif

               if(discardFirst)
                  imp->impPtr(&arg);

               double avg_exec_time[MAX_EXEC_PLANS];
               for(int k=0; k<nImpls; ++k)
               {
                  arg.exec_time[k] = 0;
                  avg_exec_time[k] = 0;
               }


               for(unsigned int tr=0; tr<TRAINING_RUNS; ++tr)
               {
                  imp->impPtr(&arg);

                  for(unsigned int k=0; k<nImpls; ++k)
                  {
                     avg_exec_time[k] += arg.exec_time[k];
                  }
               }
               for(unsigned int k=0; k<nImpls; ++k)
               {
                  avg_exec_time[k] /= TRAINING_RUNS;

                  switch(imp->impType)
                  {
                  case IMPL_CPU:
                     point->cpuImpTime[k].first = imp;
                     point->cpuImpTime[k].second = avg_exec_time[k];
                     break;
                  case IMPL_OMP:
                     point->ompImpTime[k].first = imp;
                     point->ompImpTime[k].second = avg_exec_time[k];
                     break;
                  case IMPL_CUDA:
                     point->cudaImpTime[k].first = imp;
                     point->cudaImpTime[k].second = avg_exec_time[k];
                     break;
                  default:
                     assert(false);
                  }

                  if(bestPointTime[k] <= 0 || avg_exec_time[k] < bestPointTime[k])
                  {
                     bestPointTime[k] = avg_exec_time[k];
                     bestPointImp[k] = imp;
                  }
               }
            }

            for(unsigned int k=0; k<nImpls; ++k)
            {
               assert(bestPointImp[k] != NULL && bestPointTime[k] > 0);
               point->bestImpTime[k].first = bestPointImp[k];
               point->bestImpTime[k].second = bestPointTime[k];
            }

            arrNodes[n]->points.push_back(point);

            // only discard first time...
            discardFirst = false;
         }

         for(unsigned int k=0; k<nImpls; ++k)
         {
            // now iterate over all points in the node to find out if further split is needed or not....
            bool isClosed = true;
            assert(arrNodes[n]->points.empty() == false);
            ImpDetail *bestNodeImp = arrNodes[n]->points[0]->bestImpTime[k].first;
            for(unsigned int i=1; i< arrNodes[n]->points.size(); ++i)
            {
               if(arrNodes[n]->points[i]->bestImpTime[k].first->impName != bestNodeImp->impName)
               {
                  isClosed = false;
                  break;
               }
            }

            /*! to check for any depth */
            // 	    isClosed = false;

            arrNodes[n]->isClosed[k] = isClosed;

            if(!isClosed) // further split needed
               arrNodes[n]->bestNodeImp[k] = NULL;
            else
               arrNodes[n]->bestNodeImp[k] = bestNodeImp;
         }
      }

      depth++;

      if(depth>=m_maxDepth) // break ther loop, exit condition....
         break;

      std::vector<Node*> arrNodesNew;
      for(unsigned int n=0; n<arrNodes.size(); ++n)
      {
         /*! TODO current strategy, if any (even one) combination requires more training, continue doing so */
         bool isClosed = true;
         for(unsigned int k=0; k<nImpls; ++k)
         {
            if(arrNodes[n]->isClosed[k] == false)
            {
               isClosed = false;
               break;
            }
         }
         // for each open node, generate childs....
         if(isClosed == false)
         {
            std::vector<size_t> &lowBound = arrNodes[n]->lowBounds;
            std::vector<size_t> &uppBound = arrNodes[n]->uppBounds;

            if((lowBound[0]<(uppBound[0]-2)))
            {
               // generate child nodes for arrNodes[n]
               generateSubSpaces(lowBound, uppBound, arrNodes[n]);

               for(unsigned int c=0; c<arrNodes[n]->childs.size(); ++c)
               {
                  arrNodesNew.push_back(arrNodes[n]->childs[c]);
               }
            }
            else
            {
               std::cerr << "*** Too small dimension gap to partition further....\n";
               break;
            }
         }
      }

      if(arrNodesNew.empty()) // break ther loop, exit condition....
         break;

      arrNodes = arrNodesNew;
   }
}






/*!
 * Thie method generates subspaces for an n-dimentional subspace by recursively dividing each dimension into 2 halves.
 */
void Trainer::generateSubSpaces(std::vector<size_t> &baseLowBound, std::vector<size_t> &baseUppBound, Node *father)
{
   assert(baseLowBound.empty() == false && baseLowBound.size() == baseUppBound.size() && father != NULL);

   std::vector<Node*> &childs = father->childs;

   unsigned int dimens = baseLowBound.size();

   size_t **ranges = new size_t*[dimens];
   for(unsigned int i=0; i<dimens; ++i)
      ranges[i]=new size_t[4];

   size_t upp, low, midLow, midUpp;
   for(unsigned int i=0; i<dimens; ++i)
   {
      low = baseLowBound[i];
      upp =  baseUppBound[i];

      assert(low < (upp-2));

      midLow = (low + upp) / 2;
      midUpp = midLow+1;

      DEBUG_TUNING_LEVEL3 ("Subspaces: " << low << " - " << midLow << " AND " << midUpp << " - " << upp << "\n");

      ranges[i][0]=low;
      ranges[i][1]=midLow;
      ranges[i][2]=midUpp;
      ranges[i][3]=upp;
   }


   int *binFlagArr = new int[dimens];
   for(int i=0; i<dimens; ++i)
   {
      binFlagArr[i] = 0;
   }

   bool more = false;
   std::vector<size_t> lowBounds,uppBounds;
   do
   {
      uppBounds.clear();
      lowBounds.clear();

      for(unsigned int x=0; x<dimens; ++x)
      {
         if(binFlagArr[x] == 0)
         {
            lowBounds.push_back(ranges[x][0]);
            uppBounds.push_back(ranges[x][1]);
         }
         else
         {
            lowBounds.push_back(ranges[x][2]);
            uppBounds.push_back(ranges[x][3]);
         }
      }


      Node *node = new Node(lowBounds, uppBounds, father->nImpls);
      node->id = treeNodeId++;
      node->father = father;

      node->level = (father->level + 1);

      childs.push_back(node);

//       DEBUG_TUNING_LEVEL3("--------------------------------\n");
//       DEBUG_TUNING_LEVEL3("lowBounds: " << lowBounds);
//       DEBUG_TUNING_LEVEL3("uppBounds: " << uppBounds);

      // get next binary flag...
      more = false;
      for(int i=dimens-1; i>=0; --i)
      {
         if(binFlagArr[i] == 0)
         {
            more = true;

            binFlagArr[i] = 1;

            for(int k=dimens-1; k>i; --k)
               binFlagArr[k]=0;

            break;
         }
      }

   }
   while( more );

   //free memory
   delete [] binFlagArr;

   //free memory
   for(unsigned int i=0; i<dimens; ++i)
      delete [] ranges[i];
   delete [] ranges;
}


/*!
 * This method generates all possible point combinations for two points vectors pointing to lower and upper bouund 
 * These combinations are considered the training points where different implementations are executed upon...
 */
void Trainer::generateAllPossibleCombinations(std::vector<size_t> &lowBound, std::vector<size_t> &uppBound, std::vector<std::vector<size_t> > &combinations, bool overSample)
{
//     std::cerr << "lowBound.size(): " << lowBound.size() << ", uppBound.size(): " << uppBound.size() << "\n";

   assert(lowBound.size() == uppBound.size());

   unsigned int dimens = lowBound.size();

   std::vector<size_t> comb;

   //If enable light oversampling, then add the middle point in the subspace as a vertices
   if(overSample)
   {
      //calculate the center point
      comb.clear();
      for(unsigned int i=0; i<dimens; ++i)
      {
         comb.push_back( (size_t)(std::abs(uppBound[i]+lowBound[i])/2) );
      }

      //insert the center point to the vertex
      combinations.push_back(comb);
   }

   if(lowBound.size() == 1)
   {
      combinations.push_back(lowBound);
      combinations.push_back(uppBound);
      return;
   }


   int *binFlagArr = new int[dimens];
   for(int i=0; i<dimens; ++i)
   {
      binFlagArr[i] = 0;
   }


   bool more = false;

   do
   {
      comb.clear();
      for(unsigned int i=0; i<dimens; ++i)
      {
         if(binFlagArr[i]==0)
            comb.push_back(lowBound[i]);
         else
            comb.push_back(uppBound[i]);
      }
      combinations.push_back(comb);

      // get next binary flag...
      more = false;
      for(int i=dimens-1; i>=0; --i)
      {
         if(binFlagArr[i] == 0)
         {
            more = true;

            binFlagArr[i] = 1;

            for(int k=dimens-1; k>i; --k)
               binFlagArr[k]=0;

            break;
         }
      }

   }
   while(more);

   delete [] binFlagArr;
}

} // end namespace skepu
