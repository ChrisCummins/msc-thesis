/*! \file testing.h
 *  \brief Includes various testing helpers.
 */

#ifndef TESTING_H
#define TESTING_H

#ifdef _WIN32

//     #include "src/timer_windows.h"
//     typedef TimerWindows Timer;

#else

#include "src/timer_linux.h"

namespace skepu
{
// typedef TimerLinux_GTOD Timer;
}

#endif

#include "src/data_collector.h"

#endif

