/*! \file timer_linux.h
 *  \brief Contains timer class that can be used by Linux systems.
 */

#ifndef TIMER_LINUX_H
#define TIMER_LINUX_H

#include <sys/time.h>
#include <iostream>
#include <vector>

namespace skepu
{

/*!
 *  \ingroup testing
 */

/*!
 *  \class TimerLinux_GTOD
 *
 *  \brief A class that can be used measure time on Linux systems.
 *
 *  A timer class that uses the linux function gettimeofday() to measure time. The resolution can
 *  vary on different systems but is usually a few micro seconds. Can be checked with getResolutionUs().
 *  Stores all times taken, between a start and a stop, as seperate runs and they exists until timer is reset.
 *  Both total time of all runs and average can be returned.
 */
class TimerLinux_GTOD
{

private:
   timeval timerStart;
   timeval timerEnd;



   std::vector<double> multi_time; // used for estimating multi backends.
   std::vector<double> time;
   bool record_multi;


   /*!
    *  Used to estimate when using multi-Backend based on maximization
    */
   void addMultiMaxTime()
   {
      double max = 0.0f;
      if(multi_time.empty())
         return;
      for(std::vector<double>::iterator it = multi_time.begin(); it != multi_time.end(); ++it)
      {
         if (max < *it)
            max = *it;
      }
      time.push_back( max );
      multi_time.clear();  // clear it in both cases.
   }

   /*!
    *  Used to estimate when using multi-Backend based on minimization
    */
   void addMultiMinTime()
   {
      double min = 0.0f;
      if(multi_time.empty())
         return;
      for(std::vector<double>::iterator it = multi_time.begin(); it != multi_time.end(); ++it)
      {
         if (min > *it)
            min = *it;
      }
      time.push_back( min );
      multi_time.clear();  // clear it in both cases.
   }

public:

   /*!
    *  Start timing, used when measuring multi-GPU executions.
    */
   void start_record_multi()
   {
      record_multi=true;
      multi_time.clear();  // clear it in both cases.
   }


   /*!
   *  Stop timing, used when measuring multi-GPU executions.
   */
   void stop_record_multi()
   {
      addMultiMaxTime();
      record_multi=false;
   }

   TimerLinux_GTOD()
   {
      record_multi=false;
   }

   /*!
    *  Starts the timimg.
    */
   void start()
   {
      gettimeofday(&timerStart, NULL);
   }

   /*!
    *  Stops the timimg and stores time in a vector.
    */
   void stop()
   {
      gettimeofday(&timerEnd, NULL);
      if(record_multi)
         multi_time.push_back( (((timerEnd.tv_sec  - timerStart.tv_sec) * 1000000u +  timerEnd.tv_usec - timerStart.tv_usec) / 1.e6) * 1000 );
      else
         time.push_back( (((timerEnd.tv_sec  - timerStart.tv_sec) * 1000000u +  timerEnd.tv_usec - timerStart.tv_usec) / 1.e6) * 1000 );
   }

   /*!
    *  Clears all timings taken.
    */
   void reset()
   {
      if(!record_multi)
      {
         time.clear();
      }
      multi_time.clear();  // clear it in both cases.
   }




   /*!
    *  \param run The run to get timing of.
    *
    *  \return Time for a certain run.
    */
   double getTime(int run = 0)
   {
      return time.at(run);
   }

   /*!
    *  \return Total time of all stored runs.
    */
   double getTotalTime()
   {
      double totalTime = 0.0f;

      for(std::vector<double>::iterator it = time.begin(); it != time.end(); ++it)
      {
         totalTime += *it;
      }

      return totalTime;
   }

   /*!
    *  \return Average time of all stored runs.
    */
   double getAverageTime()
   {
      double totalTime = 0.0f;

      for(std::vector<double>::iterator it = time.begin(); it != time.end(); ++it)
      {
         totalTime += *it;
      }

      return (double)(totalTime/time.size());
   }

   /*!
    *  \return Max time of all stored runs.
    */
   double getMaxTime()
   {
      double max = 0.0f;
      for(std::vector<double>::iterator it = time.begin(); it != time.end(); ++it)
      {
         if (max < *it)
            max = *it;
      }
      return max;
   }


   /*!
    *  \return Minimum time of all stored runs.
    */
   double getMinTime()
   {
      double min = 0.0f;
      for(std::vector<double>::iterator it = time.begin(); it != time.end(); ++it)
      {
         if (min > *it)
            min = *it;
      }

      return min;
   }

   /*!
    *  \return The resolution of the timer in micro seconds.
    */
   double getResolutionUs()
   {
      double result = 0.0f;
      timeval tStart;
      timeval tEnd;
      gettimeofday(&tStart, NULL);
      gettimeofday(&tEnd, NULL);
      int delay = 0;

      do
      {
         delay++;
         gettimeofday(&tStart, NULL);
         for(int i = 0; i < delay; ++i) ;
         gettimeofday(&tEnd, NULL);

         result = ((((double)tEnd.tv_sec)*1000000.0) + ((double)tEnd.tv_usec)) - ((((double)tStart.tv_sec)*1000000.0) + ((double)tStart.tv_usec));

      }
      while(result == 0);

      return result;
   }

   /*!
    *  \return Number of runs stored in timer.
    */
   int getNumTimings()
   {
      return time.size();
   }
};

}

#endif
