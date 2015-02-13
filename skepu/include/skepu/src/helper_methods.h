/*! \file helper_methods.h
 *  \brief Contains few helper methods that are used globally by different classes.
 */

/*!
*  \ingroup helpers
*/

#ifndef _HELPER_METHODS_H
#define _HELPER_METHODS_H

#include <string>
#include <algorithm>
#include <fstream>
#include <sstream>

namespace skepu
{

   
template<typename T>   
std::string convertToStr(T val)
{
   std::ostringstream ss;
   ss << val;
   return ss.str();
}

/*!
 * Method to remove leading and trailing spaces from a string.
 */
static const std::string trimSpaces(const std::string& pString, const std::string& pWhitespace = " \t")
{
   const size_t beginStr = pString.find_first_not_of(pWhitespace);
   if (beginStr == std::string::npos)
   {
      // no content
      return "";
   }

   const size_t endStr = pString.find_last_not_of(pWhitespace);
   const size_t range = endStr - beginStr + 1;

   return pString.substr(beginStr, range);
}


/*!
 * Method to get a random number between a given range.
 */
template <typename T>
inline T get_random_number(T min, T max)
{
   return (T)( rand() % (int)(max-min+1) + min); //(T)( rand() % (int)max + min );
//	return min + (T)rand()/((T)RAND_MAX/(max-min)); //(T)( rand() % (int)max + min );
}


/*!
 * Method to read text file data into a string.
 */
static std::string read_file_into_string(const std::string &filename)
{
   std::string content = "";
   std::ifstream ifs(filename.c_str());
   if(ifs.is_open())
   {
      content.assign( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()) );
   }
   return content;
}

/*!
 * Method to convert to lower case.
 */
static void toLowerCase(std::string &str)
{
   std::transform(str.begin(), str.end(),str.begin(), ::tolower);
}

/*!
 * Method to convert to upper case.
 */
static void toUpperCase(std::string &str)
{
   std::transform(str.begin(), str.end(),str.begin(), ::toupper);
}


/*!
 * Method to check whether a string starts with a given pattern.
 */
static bool startsWith(const std::string& main, const std::string& prefix)
{
   return (main.substr(0, prefix.size()) == prefix);
}



/*!
 * Method to allocate host memory of a given size. Can do pinned memory allocation if enabled.
 */
template<typename T>
void allocateHostMemory(T* &data, const size_t numElems)
{
   #if defined(SKEPU_CUDA) && defined(USE_PINNED_MEMORY)
      cudaError_t status = cudaMallocHost((void**)&data, numElems*sizeof(T));
      if (status != cudaSuccess)
      {
         SKEPU_ERROR("Error allocating pinned host memory\n");
      }
   #else
      data = new T[numElems];
      if(!data)
         SKEPU_ERROR("Memory allocation failed\n");
   #endif   
}

/*!
 * Method to deallocate host memory.
 */
template<typename T>
void deallocateHostMemory(T *data)
{
   if(!data)
      return;
   
   #if defined(SKEPU_CUDA) && defined(USE_PINNED_MEMORY)
      cudaError_t status = cudaFreeHost(data);
      if (status != cudaSuccess)
      {
         SKEPU_ERROR("Error de-allocating pinned host memory.\n");
      }
   #else
      delete[] data;
   #endif
}






}


#endif
