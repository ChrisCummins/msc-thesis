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


#if DAC_DEBUG_LEVEL >= 2
#  define DAC_DEBUG_PRINT(x) std::cout << STRLOC << "  " << x << "\n";
#else
#  define DAC_DEBUG_PRINT(x)
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


#endif // MSC_THESIS_EXERCISES_TEMPLATES_DEBUG_H_
