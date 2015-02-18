/*! \file debug.h
 *  \brief Defines a few macros that includes macros to output text when debugging. The macros use std::cerr.
 */

#ifndef DEBUG_H
#define DEBUG_H

#include <assert.h>


#ifndef SKEPU_DEBUG
#define SKEPU_DEBUG 0
#endif


#if SKEPU_TUNING_DEBUG > 0
#include <iostream>
#endif

#if SKEPU_TUNING_DEBUG > 1
#define DEBUG_TUNING_LEVEL2(text) std::cerr << "[SKEPU_TUNING_L1 " << __FILE__ << ":" << __LINE__ << "] " << text;
#else
#define DEBUG_TUNING_LEVEL2(text)
#endif

#if SKEPU_TUNING_DEBUG > 2
#define DEBUG_TUNING_LEVEL3(text) std::cerr << "[SKEPU_TUNING_L2 " << __FILE__ << ":" << __LINE__ << "] " << text;
#else
#define DEBUG_TUNING_LEVEL3(text)
#endif

#if SKEPU_DEBUG > 0
#include <iostream>
#endif

#if SKEPU_DEBUG > 0
#define DEBUG_TEXT_LEVEL1(text) std::cerr << "[SKEPU_DEBUG_L1 " << __FILE__ << ":" << __LINE__ << "] " << text;
#else
#define DEBUG_TEXT_LEVEL1(text)
#endif

#if SKEPU_DEBUG > 1
#define DEBUG_TEXT_LEVEL2(text) std::cerr << "[SKEPU_DEBUG_L2 " << __FILE__ << ":" << __LINE__ << "] " << text;
#else
#define DEBUG_TEXT_LEVEL2(text)
#endif

#if SKEPU_DEBUG > 2
#define DEBUG_TEXT_LEVEL3(text) std::cerr << "[SKEPU_DEBUG_L3 " << __FILE__ << ":" << __LINE__ << "] " << text;
#else
#define DEBUG_TEXT_LEVEL3(text)
#endif


#ifndef SKEPU_ASSERT
#define SKEPU_ASSERT(expr) assert(expr)
#endif

#define SKEPU_ERROR(text) { std::cerr << "[SKEPU_ERROR " << __FILE__ << ":" << __LINE__ << "] " << text; exit(0); }

#define SKEPU_WARNING(text) { std::cerr << "[SKEPU_WARNING " << __FILE__ << ":" << __LINE__ << "] " << text; }

#define SKEPU_EXIT() exit(0)

#ifdef __GNUC__
#define SKEPU_UNLIKELY(expr)          (__builtin_expect(!!(expr),0))
#define SKEPU_LIKELY(expr)            (__builtin_expect(!!(expr),1))
#define SKEPU_ATTRIBUTE_UNUSED        __attribute__((unused))
#define SKEPU_ATTRIBUTE_INTERNAL      __attribute__ ((visibility ("internal")))
#else
#define SKEPU_UNLIKELY(expr)          (expr)
#define SKEPU_LIKELY(expr)            (expr)
#define SKEPU_ATTRIBUTE_UNUSED
#define SKEPU_ATTRIBUTE_INTERNAL
#endif

#ifdef SKEPU_OPENCL
#define CL_CHECK_ERROR(err)  if(err != CL_SUCCESS) {std::cerr<<"Error building OpenCL program!!\n" <<err <<"\n";}
#endif







#ifdef DEBUG_UTIL

#include <fstream>
#include <sstream>

class FileWriter
{
   std::string filename;
   std::ofstream file;
public:
   FileWriter(std::string _filename): filename(_filename)
   {
      file.open(filename.c_str());
      if(!file.good())
         SKEPU_ERROR("Error while opening file for writing: " << filename);
   }
   void write(std::string line){
      file << line;
   }
   ~FileWriter() {
      try
      {
         file.close();
      }catch(...) {
         SKEPU_WARNING("A problem occurred while closing the file: " << filename);
      }
   }
};

#define OPEN_FILE(filename) FileWriter f(filename);
#define WRITE_FILE(line) f.write(line);

#endif


// class skepu_error
// {
// private:
//    std::string message;
// 
// public:
//    skepu_error(std::string m)
//    {
//       message = m;
//    }
//    inline std::string getMessage()
//    {
//       return message;
//    };
//    friend std::ostream& operator<<(std::ostream &os, skepu_error& err)
//    {
//       os<<"SKEPU: "<<(err.getMessage())<<"\n";
//       return os;
//    }
// 
// };
// 
// #ifndef _NO_EXCEPTION
// 
// #define SKEPU_ERROR(ErrormMsg)  throw skepu::skepu_error( ErrormMsg);
// 
// #else
// inline void _skepu_error (const char* pErrMsg)
// {
//    cerr << "SKEPU ERROR: " + pErrMsg << endl;
//    exit(1);
// }
// 
// #define SKEPU_ERROR(ErrormMsg)  skepu::_skepu_error( ErrormMsg);
// 
// #endif




#endif

