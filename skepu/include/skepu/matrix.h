/*! \file matrix.h
 *  \brief Contains a class declaration for the Matrix container.
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

#include <vector>
#include <map>

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
#include <cuda.h>
#endif

#include "skepu/src/malloc_allocator.h"
#include "skepu/src/environment.h"



namespace skepu
{

/*!
   *  \brief Can be used to specify whether the access is row-wise or column-wise
   *
   *  Used in some cases to mention type of access required in a certain operation.
   */
enum AccessType
{
   ROW_WISE, //C style iterating from rows
   COL_WISE // fortran style iterating from columns
};



/*!
   *  \class Matrix
   *
   *  \brief A matrix container class (2D matrix), internally uses 1D container (std::vector) to store elements in a contiguous memory allocations.
   *
   *  A \p skepu::Matrix is a 2D container that internally stores in a 1D \p std::vector to store elements in a contiguous memory allocations.
   *  Its interface and behaviour is largely compatible with \p skepu::Vector and \p std::vector but with some additions and variations.
   *  Instead of the regular element, it sometimes returns a proxy element so it can distinguish between reads
   *  and writes. It also keeps track of which parts of it are currently allocated and uploaded to the GPU.
   *  If a computation is done, changing the matrix in the GPU memory, it is not directly transferred back to the
   *  host memory. Instead, the Matrix waits until an element is accessed before any copying is done.
   *
   *  It also implements support for allocating and de-allocating page-locked memory using cudaMallocHost and cudaFreeHost.
   *  This could help is running asynchronous operations especially when using multiple CUDA devices.
   *  It can be enabled by defining USE_PINNED_MEMORY flag in the skeleton program.
   *
   *
   */
template<typename T>
class Matrix
{

   // typedefs
public:

#ifdef SKEPU_CUDA
   typedef DeviceMemPointer_CU<T>* device_pointer_type_cu;
#endif

#ifdef SKEPU_OPENCL
   typedef DeviceMemPointer_CL<T>* device_pointer_type_cl;
#endif

#ifdef USE_PINNED_MEMORY
   typedef std::vector<T, malloc_allocator<T> > container_type;
   typedef typename std::vector<T, malloc_allocator<T> >::iterator vector_iterator;
   typedef typename std::vector<T, malloc_allocator<T> >::size_type size_type;
   typedef typename std::vector<T, malloc_allocator<T> >::value_type value_type;
   typedef typename std::vector<T, malloc_allocator<T> >::difference_type difference_type;
   typedef typename std::vector<T, malloc_allocator<T> >::pointer pointer;
   typedef typename std::vector<T, malloc_allocator<T> >::reference reference;
   typedef typename std::vector<T, malloc_allocator<T> >::const_reference const_reference;
   typedef typename std::vector<T, malloc_allocator<T> >::const_iterator const_iterator;
   typedef typename std::vector<T, malloc_allocator<T> >::const_reverse_iterator const_reverse_iterator;
#else
   typedef std::vector<T> container_type;
   typedef typename std::vector<T>::iterator vector_iterator;
   typedef typename std::vector<T>::size_type size_type;
   typedef typename std::vector<T>::value_type value_type;
   typedef typename std::vector<T>::difference_type difference_type;
   typedef typename std::vector<T>::pointer pointer;
   typedef typename std::vector<T>::reference reference;
   typedef typename std::vector<T>::const_reference const_reference;
   typedef typename std::vector<T>::const_iterator const_iterator;
   typedef typename std::vector<T>::const_reverse_iterator const_reverse_iterator;
#endif

public: //-- For Testing --//

   void setValidFlag(bool val)
   {
      m_valid = val;
   }
   
   /*!
   * Get array representation
   */
   T* GetArrayRep()
   {
      return &m_data[0];
   }

   /*!
   *  \brief Overloaded stream operator, for testing purposes.
   *
   *  Outputs the matrix rowwise having one row on each line.
   */
   friend std::ostream& operator<<(std::ostream &os, Matrix<T>& matrix)
   {
      matrix.updateHost();

      os << "Matrix: ("<< matrix.total_rows() <<" X "<<matrix.total_cols()<<")\n";
      for(size_type i=0; i<matrix.size(); i++)
      {
         os<<(matrix(i))<<" ";
         if((i+1)%(matrix.total_cols())==0)
            os << "\n";
      }
      os<<"\n";
      return os;
   }


   /*!
   *  \brief Randomizes the Matrix.
   *
   *  Sets each element of the Matrix to a random number between \p min and \p max.
   *  The numbers are generated as \p integers but are cast to the type of the matrix.
   *
   *  \param min The smallest number an element can become.
   *  \param max The largest number an element can become.
   */
   void randomize(int min = 0, int max = RAND_MAX)
   {
      invalidateDeviceData();

      for(size_type i = 0; i < size(); i++)
      {
         m_data.at(i) = (T)( rand() % (int)(max-min+1) + min);
         //            m_data.at(i) = min + (T)rand()/((T)RAND_MAX/(max-min));
      }
   }

   /*!
   *  \brief Saves content of Matrix to a file.
   *
   *  Outputs the matrix as text on one line with space between elements to the specified file.
   *  Mainly for testing purposes.
   *
   *  \param filename Name of file to save to.
   */
   void save(const std::string& filename)
   {
      updateHost();

      std::ofstream file(filename.c_str());

      if (file.is_open())
      {
         for(size_type i = 0; i < m_data.size(); ++i)
         {
            file<<m_data.at(i) <<" ";
         }
         file.close();
      }
      else
      {
         std::cout<<"Unable to open file\n";
      }
   }

   /*!
   *  \brief Loads the Matrix from a file.
   *
   *  Reads a variable number of elements from a file. In the file, all elemets should be in ASCII
   *  on one line with whitespace between each element. Mainly for testing purposes.
   *
   *  \param filename Name of file to save to.
   *  \param rowWidth The width of a row. All rows get same amount of width.
   *  \param numRows The number of rows to be loaded. Default value 0 means all rows.
   */
   void load(const std::string& filename, size_type rowWidth, size_type numRows = 0)
   {
      invalidateDeviceData();

      std::ifstream file(filename.c_str());

      if (file.is_open())
      {
         std::string line;
         getline (file,line);
         std::istringstream ss(line);
         T num;
         clear();

         //Load all elements
         if(numRows == 0)
         {
            while(ss >> num)
            {
               push_back(num);
            }
         }
         // Load only numElements elements
         else
         {
            for(size_type i = 0; i < (numRows*rowWidth); ++i)
            {
               ss >> num;
               push_back(num);
            }
         }

         m_cols = rowWidth;
         m_rows = (size()/rowWidth);

         file.close();
      }
      else
      {
         std::cout<<"Unable to open file\n";
      }
   }


// Constructors, destructors
public:

   /*!
      *  Destructor, used to deallocate memory mainly, device memory.
      */
   ~Matrix()
   {
#ifdef SKEPU_OPENCL
      releaseDeviceAllocations_CL();
#endif

#ifdef SKEPU_CUDA
      releaseDeviceAllocations_CU();
#endif

      if(m_transpose_matrix)
         delete m_transpose_matrix;
   }


   /*!
      *  Constructor, used to allocate memory ($_rows * _cols$).
      * \param _rows Number of rows in the matrix.
      * \param _cols Number of columns in the matrix.
      */
   Matrix(size_type _rows, size_type _cols): m_rows(_rows), m_cols(_cols), m_data(m_rows * m_cols), m_dataChanged(false), m_transpose_matrix(0), m_noValidDeviceCopy(true), m_valid(true)
   {
#ifdef SKEPU_OPENCL
      m_transposeKernels_CL = &(Environment<T>::getInstance()->m_transposeKernels_CL);
#endif
   }

   /*!
      *  Constructor, used to allocate memory ($_rows * _cols$). With a value ot initialize all elements.
      * \param _rows Number of rows in the matrix.
      * \param _cols Number of columns in the matrix.
      * \param val A value to initialize all elements.
      */
   Matrix(size_type _rows, size_type _cols, const T& val): m_rows(_rows), m_cols(_cols),m_data(m_rows * m_cols, val), m_dataChanged(false), m_transpose_matrix(0), m_noValidDeviceCopy(true), m_valid(true)
   {
#ifdef SKEPU_OPENCL
      m_transposeKernels_CL = &(Environment<T>::getInstance()->m_transposeKernels_CL);
#endif
   }


   /*!
      *  Copy Constructor, used to assign copy of another matrix.
      * \param copy Matrix that is being assigned.
      *
      * Update the matrix before assigning it to assign latest copy.
      */
   Matrix(const Matrix<T>& copy): m_noValidDeviceCopy(true), m_valid(true)
   {
      copy.updateHost();
      this->m_rows = copy.m_rows;
      this->m_cols = copy.m_cols;
      this->m_data= copy.m_data;
      this->m_transpose_matrix = copy.m_transpose_matrix;
      this->m_dataChanged = copy.m_dataChanged;
      
#ifdef SKEPU_OPENCL
      this->m_transposeKernels_CL = copy.m_transposeKernels_CL;
#endif
   }

private:
   Matrix(): m_rows(0), m_cols(0),m_data(), m_dataChanged(false), m_transpose_matrix(0), m_noValidDeviceCopy(true), m_valid(true)
   {}

public:

   /*!
      * Returns total size of Matrix.
      * \return size of the Matrix.
      */
   size_type size() const
   {
      return m_data.size();
   }

   /*!
      * Returns total number of rows in the Matrix.
      * \return rows in the Matrix.
      */
   size_type total_rows() const
   {
      return m_rows;
   }

   /*!
      * Returns total number of columns in the Matrix.
      * \return columns in the Matrix.
      */
   size_type total_cols() const
   {
      return m_cols;
   }

   // highly dangerous, use with care.
   T *getAddress()
   {
      return &m_data[0];
   }

   /*!
      *  A small utility to change rows and columns numbers with each other. A Matrix (4x7) will become (7x4) after this function call without
      *  changing the actual values. Not similar to transpose where you actually change the values.
      */
   void change_layout()
   {
      size_type tmp = m_rows;
      m_rows=m_cols;
      m_cols = tmp;

      if(m_transpose_matrix && m_transpose_matrix->total_rows()==m_cols && m_transpose_matrix->total_cols()==m_rows && !m_dataChanged)
         m_transpose_matrix->change_layout();
   }

private:
#ifdef SKEPU_CUDA
//    std::map<std::pair< int, std::pair< T*, size_type > >, device_pointer_type_cu > m_deviceMemPointers_CU;
   std::map<std::pair< T*, size_type >, device_pointer_type_cu > m_deviceMemPointers_CU[MAX_GPU_DEVICES];

   /*! This is a temporary list that keeps track of copies that are changed on device but are not synced with host memory... */
   mutable std::map<std::pair< T*, size_type >, device_pointer_type_cu > m_deviceMemPointers_Modified_CU[MAX_GPU_DEVICES];
#endif

#ifdef SKEPU_OPENCL
   std::map<std::pair< cl_device_id, std::pair< T*, size_type > >, device_pointer_type_cl > m_deviceMemPointers_CL;
#endif

   size_type m_rows, m_cols;
   bool m_dataChanged;
   bool m_noValidDeviceCopy;


#ifdef USE_PINNED_MEMORY
   mutable std::vector<T, malloc_allocator<T> > m_data;
#else
   mutable std::vector<T> m_data;
#endif

   mutable bool m_valid; /*! to keep track of whether the main copy is valid or not */

   // for col_iterator,
   mutable Matrix<T> *m_transpose_matrix;

   template<typename Type>
   void item_swap(Type &t1, Type &t2);


// External classes
public:
   class iterator;

   class proxy_elem;

public: //-- Operators --//

   void resize(size_type _rows, size_type _cols, T val = T());

   Matrix<T>& operator=(const Matrix<T>& other);
   Matrix<T>& operator=(const T& elem);

   bool operator==(const Matrix<T>& c1);
   bool operator!=(const Matrix<T>& c1);
   bool operator<(const Matrix<T>& c1);
   bool operator>(const Matrix<T>& c1);
   bool operator<=(const Matrix<T>& c1);
   bool operator>=(const Matrix<T>& c1);

   Matrix<T>& subsection(size_type row, size_type col, size_type rowWidth, size_type colWidth);

// #if SKEPU_DEBUG>0      
   std::string m_nameVerbose; // for debugging useful
// #endif   

public: //-- STL vector regular interface --//

   //Iterators
   iterator begin();
   const_iterator begin() const;
   iterator begin(unsigned row);
   const_iterator begin(unsigned row) const;

   iterator end();
   const_iterator end() const;
   iterator end(unsigned row);
   const_iterator end(unsigned row) const;

   //Capacity
   size_type capacity() const;

   void flush();
   bool empty() const;

   //Element access
   proxy_elem at(size_type row, size_type col);
   const T& at(size_type row, size_type col) const;

   size_type row_back(size_type row);
   const T& row_back(size_type row) const;

   size_type row_front(size_type row);
   const T& row_front(size_type row) const;

   proxy_elem col_back(size_type col);
   const T& col_back(size_type col) const;

   proxy_elem col_front(size_type col);
   const T& col_front(size_type col) const;

   void clear();

   iterator erase( iterator loc );
   iterator erase( iterator start, iterator end );

   void swap(Matrix<T>& from);

public: //-- Additions to interface --//

#ifdef SKEPU_OPENCL
   device_pointer_type_cl updateDevice_CL(T* start, size_type rows, size_type cols, Device_CL* device, bool copy);
   device_pointer_type_cl updateDevice_CL(T* start, size_type cols, Device_CL* device, bool copy);
   void flush_CL();
#endif

#ifdef SKEPU_CUDA
   void copyDataToAnInvalidDeviceCopy(DeviceMemPointer_CU<T> *copy, unsigned int deviceID);
   device_pointer_type_cu updateDevice_CU(T* start, size_type rows, size_type cols, unsigned int deviceID, bool copy, bool writeAccess, bool usePitch, bool markOnlyLocalCopiesInvalid=false);
   device_pointer_type_cu updateDevice_CU(T* start, size_type cols, unsigned int deviceID, bool copy, bool writeAccess, bool markOnlyLocalCopiesInvalid=false);
   void flush_CU();
#endif

   // Care about device data
   const T& operator()(const size_type row, const size_type col) const;

   // Care about device data
   T& operator()(const size_type row, const size_type col);

   // Does not care about device data, use with care
   T& operator()(const size_type index);

   // Care about device data
   const T& operator[](const size_type index) const;

   // Care about device data
   T& operator[](const size_type index);


/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\    
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
/////////////////////////--------------------------------------------------------------------\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


   void transpose_CPU();

#ifdef SKEPU_OPENMP
   void transpose_OMP();
#endif

#ifdef SKEPU_CUDA
   void transpose_CU(Device_CU *device);
#endif

#ifdef SKEPU_OPENCL
   void transpose_CL(unsigned int deviceID);
   std::vector<std::pair<cl_kernel, Device_CL*> > *m_transposeKernels_CL;
#endif

   // unary transpose operator
   inline Matrix<T>& operator~()
   {
      if(m_transpose_matrix && m_transpose_matrix->m_rows==m_cols && m_transpose_matrix->m_cols==m_rows && !m_dataChanged)
         return *m_transpose_matrix;

#if defined(SKEPU_CUDA)
      transpose_CU(Environment<int>::getInstance()->m_devices_CU.at(Environment<int>::getInstance()->bestCUDADevID));
#elif  defined(SKEPU_OPENCL)
      transpose_CL(0);
#elif defined(SKEPU_OPENMP)
      transpose_OMP();
#else
      transpose_CPU();
#endif

      m_dataChanged = false;

      return *m_transpose_matrix;
   }

   // To be able to explicitly force updates without flushing entire matrix.
   // Could be used with operator () above to avoid unneccesary function calls
   // due to implicit synch.

   void updateHost() const;
   void invalidateDeviceData();
   void updateHostAndInvalidateDevice();
   void releaseDeviceAllocations();
   void updateHostAndReleaseDeviceAllocations();


   const Matrix<T>& operator+=(const Matrix<T>& rhs);
   const Matrix<T>& operator+=(const T& rhs);

   const Matrix<T>& operator-=(const Matrix<T>& rhs);
   const Matrix<T>& operator-=(const T& rhs);

   const Matrix<T>& operator*=(const Matrix<T>& rhs);
   const Matrix<T>& operator*=(const T& rhs);

   const Matrix<T>& operator/=(const Matrix<T>& rhs);
   const Matrix<T>& operator/=(const T& rhs);

   const Matrix<T>& operator%=(const Matrix<T>& rhs);
   const Matrix<T>& operator%=(const T& rhs);

private:

#ifdef SKEPU_OPENCL
   void updateHost_CL() const;
   void invalidateDeviceData_CL();
   void releaseDeviceAllocations_CL();
#endif

#ifdef SKEPU_CUDA
   void updateHost_CU(int deviceID = -1) const;
   void invalidateDeviceData_CU(int deviceID = -1);
   void releaseDeviceAllocations_CU(int deviceID = -1);
   
   bool isModified_CU(unsigned int deviceID);
#endif

}; // end class Matrix...


} // end namespace skepu

#include "src/matrix_iterator.inl"

#include "src/matrix_proxy.inl"
#include "src/matrix.inl"

#include "src/matrix_transpose.inl"


#ifdef SKEPU_OPENCL
#include "src/matrix_cl.inl"
#endif

#ifdef SKEPU_CUDA
#include "src/matrix_cu.inl"
#endif

#endif


