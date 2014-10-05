#ifndef EXERCISES_TEMPLATES_DAC_TIMER_H_
#define EXERCISES_TEMPLATES_DAC_TIMER_H_

#include <sys/time.h>
#include <ctime>

class Timer {
 public:
     int us();
     int ms();
     void reset();
     Timer() { reset(); }
 private:
     struct timeval start;
};

#endif  // EXERCISES_TEMPLATES_DAC_TIMER_H_
