#include "timer.h"

void Timer::reset() {
    gettimeofday(&this->start, NULL);
}

long int Timer::us() {
    struct timeval now;
    gettimeofday(&now, NULL);

    return ((now.tv_sec * 1000000 + now.tv_usec) -
            (start.tv_sec * 1000000 + start.tv_usec));
};

long int Timer::ms() {
    return us() / 1000;
};
