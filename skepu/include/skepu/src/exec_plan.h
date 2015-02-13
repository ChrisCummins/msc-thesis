/*! \file exec_plan.h
 *  \brief Contains a class that stores information about which back ends to use when executing.
 */

#ifndef EXEC_PLAN_H
#define EXEC_PLAN_H

#include <map>


namespace skepu
{


/*!
 *  \brief Can be used to specify properties for a backend.
 *
 */
struct BackEndParams
{
   BackEndParams(): backend(CPU_BACKEND), maxThreads(256), maxBlocks(65535), numOmpThreads(1)  {}
   BackEnd backend;
   size_t maxThreads;
   size_t maxBlocks;
   unsigned int numOmpThreads;
};


/*!
 *  \ingroup tuning
 */

/*!
 *  \class ExecPlan
 *
 *  \brief A class that describes an execution plan
 *
 *  This class is used to specifiy execution parameters. For the GPU back ends
 *  you can set both the block size (maxThreads) and the grid size (maxBlocks).
 *  For OpenMP the number of threads is parameterized (numOmpThreads).
 *
 *  It is also possible to specify which backend should be used for a certain
 *  data size. This is done by adding a lowBound and a highBound of data sizes
 *  and a backend that should be used for that range to a list. The skeletons
 *  will use this list when deciding which back end to use.
 */
class ExecPlan
{
public:
   ExecPlan()
   {
      m_cacheEntry.first = 0;
      calibrated = false;
   }

   /*! boolean field to specify if this exec plan is properly initialized or not */
   bool calibrated;

   void add(size_t lowBound, size_t highBound, BackEnd backend, size_t gs, size_t bs)
   {
      BackEndParams bp;
      bp.backend=backend;
      bp.maxThreads=bs;
      bp.maxBlocks = gs;
      sizePlan.insert(std::make_pair(std::make_pair(lowBound, highBound),bp));
   }

   void add(size_t lowBound, size_t highBound, BackEnd backend, unsigned int numOmpThreads)
   {
      BackEndParams bp;
      bp.backend=backend;
      bp.numOmpThreads = numOmpThreads;
      sizePlan.insert(std::make_pair(std::make_pair(lowBound, highBound),bp));
   }

   void add(size_t lowBound, size_t highBound, BackEndParams params)
   {
      sizePlan.insert(std::make_pair(std::make_pair(lowBound, highBound),params));
   }

   void add(size_t lowBound, size_t highBound, BackEnd backend)
   {
      BackEndParams bp;
      bp.backend=backend;
      sizePlan.insert(std::make_pair(std::make_pair(lowBound, highBound),bp));
   }

   void setMaxThreads(size_t size, size_t maxthreads)
   {
      std::map< std::pair<size_t, size_t>, BackEndParams >::iterator it;
      for(it = sizePlan.begin(); it != sizePlan.end(); ++it)
      {
         if(size >= it->first.first && size <= it->first.second)
         {
            it->second.maxThreads=maxthreads;
            return;
         }
      }

      (--it)->second.maxThreads= maxthreads;
   }

   void setMaxBlocks(size_t size, size_t maxBlocks)
   {
      std::map< std::pair<size_t, size_t>, BackEndParams >::iterator it;
      for(it = sizePlan.begin(); it != sizePlan.end(); ++it)
      {
         if(size >= it->first.first && size <= it->first.second)
         {
            it->second.maxBlocks=maxBlocks;
            return;
         }
      }

      (--it)->second.maxBlocks= maxBlocks;
   }

   void setNumOmpThreads(size_t size, unsigned int ompThreads)
   {
      std::map< std::pair<size_t, size_t>, BackEndParams >::iterator it;
      for(it = sizePlan.begin(); it != sizePlan.end(); ++it)
      {
         if(size >= it->first.first && size <= it->first.second)
         {
            it->second.numOmpThreads = ompThreads;
            return;
         }
      }

      (--it)->second.numOmpThreads = ompThreads;
   }

   unsigned int numOmpThreads(size_t size)
   {
      if(sizePlan.empty())
         return 1;

      std::map< std::pair<size_t, size_t>, BackEndParams >::iterator it;
      for(it = sizePlan.begin(); it != sizePlan.end(); ++it)
      {
         if(size >= it->first.first && size <= it->first.second)
         {
            return it->second.numOmpThreads;
         }
      }

      return (--it)->second.numOmpThreads;
   }

   size_t maxThreads(size_t size)
   {
      if(sizePlan.empty())
         return 32;

      std::map< std::pair<size_t, size_t>, BackEndParams >::iterator it;
      for(it = sizePlan.begin(); it != sizePlan.end(); ++it)
      {
         if(size >= it->first.first && size <= it->first.second)
         {
            return it->second.maxThreads;
         }
      }

      return (--it)->second.maxThreads;
   }

   bool isTrainedFor(size_t size)
   {
      if(sizePlan.empty())
         return false;

      std::map< std::pair<size_t, size_t>, BackEndParams >::iterator it;
      for(it = sizePlan.begin(); it != sizePlan.end(); ++it)
      {
         if(size >= it->first.first && size <= it->first.second)
         {
            return true;
         }
      }

      return false;
   }

   BackEnd find(size_t size)
   {
      if(sizePlan.empty())
         return CPU_BACKEND;

      if(m_cacheEntry.first == size)
         return m_cacheEntry.second.backend;

      std::map< std::pair<size_t, size_t>, BackEndParams >::iterator it;
      for(it = sizePlan.begin(); it != sizePlan.end(); ++it)
      {
         if(size >= it->first.first && size <= it->first.second)
         {
            m_cacheEntry = std::make_pair(size, it->second);
            return m_cacheEntry.second.backend;
         }
      }

      m_cacheEntry = std::make_pair(size, (--it)->second);
      return m_cacheEntry.second.backend;
   }

   BackEndParams find_(size_t size)
   {
      if(sizePlan.empty())
         return BackEndParams();

      if(m_cacheEntry.first == size)
         return m_cacheEntry.second;

      std::map< std::pair<size_t, size_t>, BackEndParams >::iterator it;
      for(it = sizePlan.begin(); it != sizePlan.end(); ++it)
      {
         if(size >= it->first.first && size <= it->first.second)
         {
            m_cacheEntry = std::make_pair(size, it->second);
            return m_cacheEntry.second;
// 		return it->second;
         }
      }

      m_cacheEntry = std::make_pair(size, (--it)->second);
      return m_cacheEntry.second;
// 	    return (--it)->second;
   }

   void clear()
   {
      sizePlan.clear();
   }


public:
   std::pair<size_t, BackEndParams> m_cacheEntry;
   std::map< std::pair<size_t, size_t>, BackEndParams > sizePlan;
};

}

#endif

