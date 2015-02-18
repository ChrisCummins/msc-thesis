/*! \file vector.h
 *  \brief Contains a class declaration for the Vector container.
 */

#ifndef VECTOR_H
#define VECTOR_H

#include <fstream>
#include <sstream>

#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include <cstddef>

#include <map>

#include "skepu/matrix.h"
#include "skepu/src/malloc_allocator.h"
#include "skepu/src/environment.h"

#ifdef SKEPU_OPENCL
#ifdef USE_MAC_OPENCL
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "src/device_mem_pointer_cl.h"
#endif

#ifdef SKEPU_CUDA
#include "src/device_mem_pointer_cu.h"
#endif


namespace skepu
{

/*!
 *  \class Vector
 *
 *  \brief A vector container class, implemented as a wrapper for std::vector.
 *
 *  A \p skepu::Vector is a container of vector/array type and is implemented as a wrapper for \p std::vector.
 *  Its interface and behaviour is largely compatible with \p std::vector but with some additions and variations.
 *  Instead of the regular element, it sometimes returns a proxy element so it can distinguish between reads
 *  and writes. It also keeps track of which parts of it are currently allocated and uploaded to the GPU.
 *  If a computation is done, changing the vector in the GPU memory, it is not directly transferred back to the
 *  host memory. Instead, the vector waits until an element is accessed before any copying is done.
 *
 *  It also implements support for allocating and de-allocating page-locked memory using cudaMallocHost and cudaFreeHost.
 *  This could help is running asynchronous operations especially when using multiple CUDA devices.
 *  It can be enabled by defining USE_PINNED_MEMORY flag in the skeleton program.
 *
 *  Please refer to C++ STL vector documentation for more information about CPU side implementation.
 */
template <typename T>
class Vector
{

public:

   typedef typename std::vector<T>::size_type size_type;
   typedef T value_type;
   typedef ptrdiff_t difference_type;
   typedef T* pointer;
   typedef T& reference;
   typedef T const & const_reference;

   //-- For Testing --//

   /*!
    *  \brief Overloaded stream operator, for testing purposes.
    *
    *  Outputs the vector on one line with space between elements to the chosen stream.
    */
   friend std::ostream& operator<< (std::ostream& output, Vector<T>& vec)
   {
      for(size_type i = 0; i < vec.size(); ++i)
      {
         output<<vec.at(i) <<" ";
      }

      return output;
   }

public: //-- For Testing --//

   /*!
    *  \brief Randomizes the vector.
    *
    *  Sets each element of the vector to a random number between \p min and \p max.
    *  The numbers are generated as \p integers but are cast to the type of the vector.
    *
    *  \param min The smallest number an element can become.
    *  \param max The largest number an element can become.
    */
   void randomize(int min = 0, int max = RAND_MAX)
   {
      invalidateDeviceData();

      for(size_type i = 0; i < size(); i++)
      {
         m_data[i] = (T)( rand() % max + min );
      }
   }

   /*!
    *  \brief Saves content of vector to a file.
    *
    *  Outputs the vector as text on one line with space between elements to the specified file.
    *  Mainly for testing purposes.
    *
    *  \param filename Name of file to save to.
    */
   void save(const std::string& filename)
   {
      std::ofstream file(filename.c_str());

      if (file.is_open())
      {
         for(size_type i = 0; i < size(); ++i)
         {
            file<<at(i) <<" ";
         }
         file.close();
      }
      else
      {
         std::cout<<"Unable to open file\n";
      }
   }

   /*!
    *  \brief Loads the vector from a file.
    *
    *  Reads a variable number of elements from a file. In the file, all elemets should be in ASCII
    *  on one line with whitespace between each element. Mainly for testing purposes.
    *
    *  \param filename Name of file to save to.
    *  \param numElements The number of elements to load. Default value 0 means all values.
    */
   void load(const std::string& filename, size_type numElements = 0)
   {
      std::ifstream file(filename.c_str());

      if (file.is_open())
      {
         std::string line;
         getline (file,line);
         std::istringstream ss(line);
         T num;
         clear();

         //Load all elements
         if(numElements == 0)
         {
            while(ss >> num)
            {
               push_back(num);
            }
         }
         // Load only numElements elements
         else
         {
            for(size_type i = 0; i < numElements; ++i)
            {
               ss >> num;
               push_back(num);
            }
         }

         file.close();
      }
      else
      {
         std::cout<<"Unable to open file\n";
      }
   }

public: //-- Typedefs --//


#ifdef SKEPU_CUDA
   typedef DeviceMemPointer_CU<T>* device_pointer_type_cu;
#endif

#ifdef SKEPU_OPENCL
   typedef DeviceMemPointer_CL<T>* device_pointer_type_cl;
#endif




public: //-- Constructors & Destructor --//

   Vector();

   Vector(const Vector& vec);

   explicit Vector(size_type num, const T& val = T());

   Vector(T * const ptr, size_type size, bool deallocEnabled = true);

   ~Vector();

public: //-- Member classes --//

   class iterator;

   class proxy_elem;

public: //-- Operators --//

   proxy_elem operator[](const size_type index);

   const T& operator[](const size_type index) const;

   Vector<T>& operator=(const Vector<T>& other);

   bool operator==(const Vector<T>& c1);
   bool operator!=(const Vector<T>& c1);

   bool operator<(const Vector<T>& c1);
   bool operator>(const Vector<T>& c1);
   bool operator<=(const Vector<T>& c1);
   bool operator>=(const Vector<T>& c1);

public: //-- STL vector regular interface --//

   //Iterators
   iterator begin();

   iterator end();

   //Capacity
   size_type capacity() const;

   size_type size() const;

   size_type max_size() const;

   void resize(size_type num, T val = T());

   bool empty() const;

   void reserve(size_type size);


   //Element access
   proxy_elem at(size_type loc);
   const T& at(size_type loc) const;

   proxy_elem back();
   const T& back() const;

   proxy_elem front();
   const T& front() const;


   //Modifiers
   void assign( size_type num, const T& val );

   template <typename input_iterator>
   void assign( input_iterator start, input_iterator end );

   void clear();

   iterator erase( iterator loc );
   iterator erase( iterator start, iterator end );

   iterator insert( iterator loc, const T& val );

   void insert( iterator loc, size_type num, const T& val );

   void pop_back();

   void push_back(const T& val);

   void swap(Vector<T>& from);

   T *getAddress()
   {
      return m_data;
   }

public: //-- Additions to interface --//



#ifdef SKEPU_OPENCL
   device_pointer_type_cl updateDevice_CL(T* start, size_type numElements, Device_CL* device, bool copy);
   void flush_CL();
   bool isVectorOnDevice_CL(Device_CL* device, bool multi=false);
#endif

#ifdef SKEPU_CUDA
   void copyDataToAnInvalidDeviceCopy(DeviceMemPointer_CU<T> *copy, unsigned int deviceID);
   device_pointer_type_cu updateDevice_CU(T* start, size_type numElements, unsigned int deviceID, bool copy, bool writeAccess, bool markOnlyLocalCopiesInvalid = false);
   void flush_CU();
   bool isVectorOnDevice_CU(unsigned int deviceID);
   bool isModified_CU(unsigned int deviceID);
#endif

   void flush();

   // Does not care about device data, use with care
   T& operator()(const size_type index);

   const T& operator()(const size_type index) const;

   // To be able to explicitly force updates without flushing entire vector.
   // Could be used with operator () above to avoid unneccesary function calls
   // due to implicit synch.
   void updateHost() const;
   void invalidateDeviceData();
   void updateHostAndInvalidateDevice();
   void releaseDeviceAllocations();
   void updateHostAndReleaseDeviceAllocations();

// #if SKEPU_DEBUG>0      
   std::string m_nameVerbose; // for debugging useful
// #endif

   void setValidFlag(bool val)
   {
      m_valid = val;
   }

private: //-- Data --//
   T *m_data;
   mutable bool m_valid; /*! to keep track of whether the main copy is valid or not */
   size_type m_capacity;
   size_type m_size;
   bool m_deallocEnabled;
   bool m_noValidDeviceCopy;


#ifdef SKEPU_OPENCL
   std::map<std::pair< cl_device_id, T* >, device_pointer_type_cl > m_deviceMemPointers_CL;
#endif

#ifdef SKEPU_CUDA
//      std::map<std::pair< int, std::pair< T*, size_type > >, device_pointer_type_cu > m_deviceMemPointers_CU;
   std::map<std::pair< T*, size_type >, device_pointer_type_cu > m_deviceMemPointers_CU[MAX_GPU_DEVICES];

   /*! This is a temporary list that keeps track of copies that are changed on device but are not synced with host memory... */
   mutable std::map<std::pair< T*, size_type >, device_pointer_type_cu > m_deviceMemPointers_Modified_CU[MAX_GPU_DEVICES];
#endif

//-- Private helpers --//

#ifdef SKEPU_OPENCL
   void updateHost_CL() const;
   void invalidateDeviceData_CL();
   void releaseDeviceAllocations_CL();
#endif

#ifdef SKEPU_CUDA
   void updateHost_CU(int deviceID = -1) const;
   void invalidateDeviceData_CU(int deviceID = -1);
   void releaseDeviceAllocations_CU(int deviceID = -1);
#endif



};

}

#include "src/vector_iterator.inl"
#include "src/vector_proxy.inl"
#include "src/vector.inl"



#ifdef SKEPU_OPENCL
#include "src/vector_cl.inl"
#endif

#ifdef SKEPU_CUDA
#include "src/vector_cu.inl"
#endif



#endif

