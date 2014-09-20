#include "timer.h"

void Timer::reset() {
    gettimeofday(&this->start, NULL);
}

unsigned int Timer::us() {
    struct timeval now;
    gettimeofday(&now, NULL);

    return now.tv_usec - start.tv_usec;
};

unsigned int Timer::ms() {
    return us() / 1000;
};
