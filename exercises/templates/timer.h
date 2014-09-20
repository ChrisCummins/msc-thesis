#ifndef MSC_THESIS_EXERCISES_TEMPLATES_TIMER_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_TIMER_H_

#include <sys/time.h>
#include <ctime>

class Timer {
 public:
     long int us();
     long int ms();
     void reset();
     Timer() { reset(); }
 private:
     struct timeval start;
};

#endif // MSC_THESIS_EXERCISES_TEMPLATES_TIMER_H_
