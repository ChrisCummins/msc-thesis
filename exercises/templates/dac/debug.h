#ifndef MSC_THESIS_EXERCISES_TEMPLATES_DEBUG_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_DEBUG_H_

#include <assert.h>

#define STRLOC "" << __FILE__ << ":" << __LINE__ << " " << __func__ << "()"


#ifdef DAC_DEBUG_LEVEL
#  define DAC_DEBUG
#endif


#ifdef DAC_DEBUG
#  ifndef DAC_DEBUG_LEVEL
#    define DAC_DEBUG_LEVEL 1
#  endif
#endif

#ifdef DAC_DEBUG
#  define IF_DAC_DEBUG(x) x
#  define IFN_DAC_DEBUG(x)
#  define DAC_ASSERT(x) assert(x)
#else
#  define IF_DAC_DEBUG(x)
#  define IFN_DAC_DEBUG(x) x
#  define DAC_ASSERT(x)
#endif

/************************/
/* Debugging print outs */
/************************/

#ifdef DAC_DEBUG
#  define DAC_DEBUG_PRINT(level, msg) DAC_DEBUG_PRINT_##level(msg)
#  define _DAC_DEBUG_PRINT(level, msg) \
    std::cout << "[DEBUG " << level << " " << STRLOC << "]  " << msg << "\n";
#else
#  define DAC_DEBUG_PRINT(level, msg)
#  define _DAC_DEBUG_PRINT(level, msg)
#endif

#if DAC_DEBUG_LEVEL >= 1
#  define DAC_DEBUG_PRINT_1(msg) _DAC_DEBUG_PRINT(1, msg)
#else
#  define DAC_DEBUG_PRINT_1(msg)
#endif

#if DAC_DEBUG_LEVEL >= 2
#  define DAC_DEBUG_PRINT_2(msg) _DAC_DEBUG_PRINT(2, msg)
#else
#  define DAC_DEBUG_PRINT_2(msg)
#endif

#if DAC_DEBUG_LEVEL >= 3
#  define DAC_DEBUG_PRINT_3(msg) _DAC_DEBUG_PRINT(3, msg)
#else
#  define DAC_DEBUG_PRINT_3(msg)
#endif

#endif // MSC_THESIS_EXERCISES_TEMPLATES_DEBUG_H_
