/*! \file device_mem_pointer_cu.h
 *  \brief Contains a class declaration for an object which represents an CUDA device memory allocation for Vector container.
 */

#ifndef DEVICE_MEM_POINTER_CU_H
#define DEVICE_MEM_POINTER_CU_H

#ifdef SKEPU_CUDA

#include <iostream>
#include <cuda.h>



#include "device_cu.h"



namespace skepu
{

/*!
 *  \ingroup helpers
 */

/*!
 *  \class DeviceMemPointer_CU
 *
 *  \brief A class representing a CUDA device memory allocation for container.
 *
 *  This class represents a CUDA device 1D memory allocation and controls the data transfers between
 *  host and device.
 */
template <typename T>
class DeviceMemPointer_CU
{

public:
   DeviceMemPointer_CU(T* start, int numElements, Device_CU *device);
   DeviceMemPointer_CU(T* start, int rows, int cols, Device_CU *device, bool usePitch=false);

   DeviceMemPointer_CU(T* root, T* start, int numElements, Device_CU *device);
   ~DeviceMemPointer_CU();

   void copyHostToDevice(int numElements = -1) const;
   void copyDeviceToHost(int numElements = -1) const;
   T* getDeviceDataPointer() const;
   int getDeviceID() const;
   void changeDeviceData();

   size_t m_pitch;

   size_t m_rows;

   size_t m_cols;

   bool deviceDataHasChanged() const
   {
      return m_deviceDataHasChanged;
   }

   // marks first initialization, useful when want to separate actual CUDA allocation and memory copy (HTD) such as when using mulit-GPU CUDA.
   mutable bool m_initialized;

private:
   T* m_hostDataPointer;
   T* m_deviceDataPointer;
   int m_numElements;
   int m_deviceID;

   Device_CU *m_dev;

   bool m_usePitch;

   mutable bool m_deviceDataHasChanged;
};

/*!
 *  The constructor allocates a certain amount of space in device memory and stores a pointer to
 *  some data in host memory.
 *
 *  \param start Pointer to data in host memory.
 *  \param numElements Number of elements to allocate memory for.
 *  \param device pointer to Device_CU object of a valid CUDA device to allocate memory on.
 */
template <typename T>
DeviceMemPointer_CU<T>::DeviceMemPointer_CU(T* start, int numElements, Device_CU *device) : m_hostDataPointer(start), m_numElements(numElements), m_dev(device), m_rows(1), m_cols(numElements), m_pitch(numElements), m_initialized(false), m_usePitch(false)
{
   cudaError_t err;
   size_t sizeVec = m_numElements*sizeof(T);

   DEBUG_TEXT_LEVEL1("Alloc: " <<m_numElements << "\n")

   m_deviceID = m_dev->getDeviceID();

   cudaSetDevice(m_deviceID);

   err = cudaMalloc((void**)&m_deviceDataPointer, sizeVec);
   if(err != cudaSuccess)
   {
      std::cerr<<"Error allocating memory on device\n";
   }

   m_deviceDataHasChanged = false;
}



/*!
 *  The constructor allocates a certain amount of space in device memory and stores a pointer to
 *  some data in host memory.
 *
 *  \param start Pointer to data in host memory.
 *  \param rows Number of rows to allocate memory for.
 *  \param cols Number of columns to allocate memory for.
 *  \param device pointer to Device_CU object of a valid CUDA device to allocate memory on.
 *  \param usePitch To specify whether to use padding to ensure proper coalescing for row-wise access from CUDA global memory.
 */
template <typename T>
DeviceMemPointer_CU<T>::DeviceMemPointer_CU(T* start, int rows, int cols, Device_CU *device, bool usePitch) : m_hostDataPointer(start), m_numElements(rows*cols), m_rows(rows), m_cols(cols), m_dev(device), m_initialized(false), m_usePitch(usePitch)
{
   cudaError_t err;
   size_t sizeVec = m_numElements*sizeof(T);

   DEBUG_TEXT_LEVEL1("Alloc: " <<m_numElements << "\n")

   m_deviceID = m_dev->getDeviceID();

   cudaSetDevice(m_deviceID);

   if(m_usePitch)
   {
      err = cudaMallocPitch((void**)&m_deviceDataPointer, &m_pitch, cols * sizeof(T), rows);
      m_pitch = (m_pitch)/sizeof(T);
   }
   else
   {
      err = cudaMalloc((void**)&m_deviceDataPointer, sizeVec);
      m_pitch = cols;
   }

   if(err != cudaSuccess)
   {
      std::cerr<<"Error allocating memory on device\n";
   }

   m_deviceDataHasChanged = false;
}


/*!
 *  The constructor allocates a certain amount of space in device memory and stores a pointer to
 *  some data in host memory.
 *
 *  \param root Pointer to starting address of data in host memory (can be same as start).
 *  \param start Pointer to data in host memory.
 *  \param numElements Number of elements to allocate memory for.
 *  \param device pointer to Device_CU object of a valid CUDA device to allocate memory on.
 */
template <typename T>
DeviceMemPointer_CU<T>::DeviceMemPointer_CU(T* root, T* start, int numElements, Device_CU *device) : m_hostDataPointer(start), m_numElements(numElements), m_dev(device), m_rows(1), m_cols(numElements), m_pitch(numElements), m_initialized(false), m_usePitch(false)
{
   cudaError_t err;
   size_t sizeVec = m_numElements*sizeof(T);

   DEBUG_TEXT_LEVEL1("Alloc: " <<m_numElements << "\n")

   m_deviceID = m_dev->getDeviceID();

   cudaSetDevice(m_deviceID);

   err = cudaMalloc((void**)&m_deviceDataPointer, sizeVec);
   if(err != cudaSuccess)
   {
      std::cerr<<"Error allocating memory on device\n";
   }

   m_deviceDataHasChanged = false;
}

/*!
 *  The destructor releases the allocated device memory.
 */
template <typename T>
DeviceMemPointer_CU<T>::~DeviceMemPointer_CU()
{
   DEBUG_TEXT_LEVEL1("DeAlloc: " <<m_numElements <<"\n")

   cudaSetDevice(m_deviceID);

   cudaFree(m_deviceDataPointer);
}

/*!
 *  Copies data from host memory to device memory.
 *
 *  \param numElements Number of elements to copy, default value -1 = all elements.
 */
template <typename T>
void DeviceMemPointer_CU<T>::copyHostToDevice(int numElements) const
{
   if(m_hostDataPointer != NULL)
   {
      DEBUG_TEXT_LEVEL1("HOST_TO_DEVICE: "<<((numElements==-1)? m_numElements: numElements)<<"!!!\n")

      cudaError_t err;
      size_t sizeVec;

      // used for pitch allocation.
      int _rows, _cols;

      if(numElements < 1)
      {
         numElements = m_numElements;
      }

      if(m_usePitch)
      {
         if( (numElements%m_cols)!=0 || (numElements/m_cols)<1 ) // using pitch option, memory copy must be proper, respecting rows and cols
         {
            std::cerr<<"Error! Cannot copy data using pitch option when size mismatches with rows and columns. numElements: "<<numElements<<",  rows:"<< m_rows <<", m_cols: "<<m_cols<<"\n";
         }

         _rows = numElements/m_cols;
         _cols = m_cols;
      }

      sizeVec = numElements*sizeof(T);

      cudaSetDevice(m_deviceID);

#ifdef USE_PINNED_MEMORY
      if(m_usePitch)
         err = cudaMemcpy2DAsync(m_deviceDataPointer,m_pitch*sizeof(T),m_hostDataPointer,_cols*sizeof(T), _cols*sizeof(T), _rows, cudaMemcpyHostToDevice, (m_dev->m_streams[0]));
      else
         err = cudaMemcpyAsync(m_deviceDataPointer, m_hostDataPointer, sizeVec, cudaMemcpyHostToDevice, (m_dev->m_streams[0]));
#else
      if(m_usePitch)
         err = cudaMemcpy2D(m_deviceDataPointer,m_pitch*sizeof(T),m_hostDataPointer,_cols*sizeof(T), _cols*sizeof(T), _rows, cudaMemcpyHostToDevice);
      else
         err = cudaMemcpy(m_deviceDataPointer, m_hostDataPointer, sizeVec, cudaMemcpyHostToDevice);
#endif

      if(err != cudaSuccess)
      {
         std::cerr<<"Error copying data to device\n" <<cudaGetErrorString(err) <<"\n";
      }

      if(!m_initialized) // set that it is initialized
         m_initialized = true;

      m_deviceDataHasChanged = false;
   }
}

/*!
 *  Copies data from device memory to host memory. Only copies if data on device has been marked as changed.
 *
 *  \param numElements Number of elements to copy, default value -1 = all elements.
 */
template <typename T>
void DeviceMemPointer_CU<T>::copyDeviceToHost(int numElements) const
{
   if(m_deviceDataHasChanged && m_hostDataPointer != NULL)
   {
      DEBUG_TEXT_LEVEL1("DEVICE_TO_HOST: "<<((numElements<1)? m_numElements: numElements)<<"!!!\n")

      cudaError_t err;
      size_t sizeVec;

      // used for pitch allocation.
      int _rows, _cols;

      if(numElements < 1)
      {
         numElements = m_numElements;
      }
      if(m_usePitch)
      {
         if( (numElements%m_cols)!=0 || (numElements/m_cols)<1 ) // using pitch option, memory copy must be proper, respecting rows and cols
         {
            std::cerr<<"Error! Cannot copy data using pitch option when size mismatches with rows and columns. numElements: "<<numElements<<",  rows:"<< m_rows <<", m_cols: "<<m_cols<<"\n";
         }

         _rows = numElements/m_cols;
         _cols = m_cols;
      }

      sizeVec = numElements*sizeof(T);

      cudaSetDevice(m_deviceID);

#ifdef USE_PINNED_MEMORY
      if(m_usePitch)
         err = cudaMemcpy2DAsync(m_hostDataPointer,_cols*sizeof(T),m_deviceDataPointer,m_pitch*sizeof(T), _cols*sizeof(T), _rows, cudaMemcpyDeviceToHost, (m_dev->m_streams[0]));
      else
      {
         err = cudaMemcpyAsync(m_hostDataPointer, m_deviceDataPointer, sizeVec, cudaMemcpyDeviceToHost, (m_dev->m_streams[0]));
      }
#else
      if(m_usePitch)
         err = cudaMemcpy2D(m_hostDataPointer,_cols*sizeof(T),m_deviceDataPointer,m_pitch*sizeof(T), _cols*sizeof(T), _rows, cudaMemcpyDeviceToHost);
      else
         err = cudaMemcpy(m_hostDataPointer, m_deviceDataPointer, sizeVec, cudaMemcpyDeviceToHost);
#endif

      if(err != cudaSuccess)
      {
         std::cerr<<"Error copying data from device: " <<cudaGetErrorString(err) <<"\n";
      }

      m_deviceDataHasChanged = false;
   }
}

/*!
 *  \return Pointer to device memory.
 */
template <typename T>
T* DeviceMemPointer_CU<T>::getDeviceDataPointer() const
{
   return m_deviceDataPointer;
}

/*!
 *  \return The device ID of the CUDA device that has the allocation.
 */
template <typename T>
int DeviceMemPointer_CU<T>::getDeviceID() const
{
   return m_deviceID;
}

/*!
 *  Marks the device data as changed.
 */
template <typename T>
void DeviceMemPointer_CU<T>::changeDeviceData()
{
   m_deviceDataHasChanged = true;
}

}

#endif

#endif

