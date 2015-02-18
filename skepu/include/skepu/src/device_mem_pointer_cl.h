/*! \file device_mem_pointer_cl.h
 *  \brief Contains a class declaration for an object which represents an OpenCL device memory allocation for container.
 */

#ifndef DEVICE_MEM_POINTER_CL_H
#define DEVICE_MEM_POINTER_CL_H

#ifdef SKEPU_OPENCL

#include <iostream>
#ifdef USE_MAC_OPENCL
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <skepu/src/environment.h>

#include "device_cl.h"


namespace skepu
{

/*!
 *  \ingroup helpers
 */

/*!
 *  \class DeviceMemPointer_CL
 *
 *  \brief A class representing an OpenCL device memory allocation for container.
 *
 *  This class represents an OpenCL device 1D memory allocation and controls the data transfers between
 *  host and device.
 */
template <typename T>
class DeviceMemPointer_CL
{

public:
   DeviceMemPointer_CL(T* root, T* start, int numElements, Device_CL* device);
   DeviceMemPointer_CL(T* start, int numElements, Device_CL* device);
   ~DeviceMemPointer_CL();

   void copyHostToDevice(int numElements = -1, bool copyLast=false) const;
   void copyDeviceToHost(int numElements = -1, bool copyLast=false) const;

   void copyDeviceToDevice(cl_mem copyToPointer,int numElements, int dstOffset = 0, int srcOffset = 0) const;

   cl_mem getDeviceDataPointer() const;
   void changeDeviceData();

   // marks first initialization, useful when want to separate actual OpenCL allocation and memory copy (HTD) such as when using mulit-GPU OpenCL.
   mutable bool m_initialized;

private:
   void copyHostToDevice_internal(T* src, cl_mem dest, int numElements, int offset=0) const;

   T* m_rootHostDataPointer;
   T* m_effectiveHostDataPointer;
   T* m_hostDataPointer;


   int m_effectiveNumElements;

   cl_mem m_deviceDataPointer;
   cl_mem m_effectiveDeviceDataPointer;

   int m_numElements;
   Device_CL* m_device;

   mutable bool deviceDataHasChanged;
};

/*!
 *  The constructor allocates a certain amount of space in device memory and stores a pointer to
 *  some data in host memory.
 *
 *  \param start Pointer to data in host memory.
 *  \param numElements Number of elements to allocate memory for.
 *  \param device Pointer to a valid device to allocate the space on.
 */
template <typename T>
DeviceMemPointer_CL<T>::DeviceMemPointer_CL(T* start, int numElements, Device_CL* device)  : m_effectiveHostDataPointer(start), m_hostDataPointer(start), m_numElements(numElements), m_effectiveNumElements(numElements), m_device(device), m_initialized(false)
{
   cl_int err;
   size_t sizeVec = numElements*sizeof(T);

   DEBUG_TEXT_LEVEL1("Alloc: " <<numElements <<"\n")
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION

#ifdef SKEPU_MEASURE_ONLY_COPY
   clFinish(m_device->getQueue());
#endif

   devMemAllocTimer.start();
#endif

   m_deviceDataPointer = clCreateBuffer(m_device->getContext(), CL_MEM_READ_WRITE, sizeVec, NULL, &err);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error allocating memory on device. size: " << sizeVec << "\n");
   }

   m_effectiveDeviceDataPointer = m_deviceDataPointer;

#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION

#ifdef SKEPU_MEASURE_ONLY_COPY
   clFinish(m_device->getQueue());
#endif

   devMemAllocTimer.stop();
#endif
   deviceDataHasChanged = false;
}



/*!
 *  The constructor allocates a certain amount of space in device memory and stores a pointer to
 *  some data in host memory. Takes a root address as well.
 *
 *  \param root Pointer to starting address of data in host memory (can be same as start).
 *  \param start Pointer to data in host memory.
 *  \param numElements Number of elements to allocate memory for.
 *  \param device Pointer to a valid device to allocate the space on.
 */
template <typename T>
DeviceMemPointer_CL<T>::DeviceMemPointer_CL(T* root, T* start, int numElements, Device_CL* device) : m_effectiveHostDataPointer(start), m_hostDataPointer(root), m_numElements(numElements), m_effectiveNumElements(numElements), m_device(device), m_initialized(false)
{
   cl_int err;
   size_t sizeVec = numElements*sizeof(T);

   DEBUG_TEXT_LEVEL1("Alloc: " <<numElements <<"\n")
#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION

#ifdef SKEPU_MEASURE_ONLY_COPY
   clFinish(m_device->getQueue());
#endif

   devMemAllocTimer.start();
#endif

   m_deviceDataPointer = clCreateBuffer(m_device->getContext(), CL_MEM_READ_WRITE, sizeVec, NULL, &err);
   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error allocating memory on device. size: " << sizeVec << "\n");
   }

   m_effectiveDeviceDataPointer = m_deviceDataPointer;

#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION

#ifdef SKEPU_MEASURE_ONLY_COPY
   clFinish(m_device->getQueue());
#endif

   devMemAllocTimer.stop();
#endif
   deviceDataHasChanged = false;
}



/*!
 *  The destructor releases the allocated device memory.
 */
template <typename T>
DeviceMemPointer_CL<T>::~DeviceMemPointer_CL()
{
   DEBUG_TEXT_LEVEL1("DeAlloc: " <<m_numElements <<"\n")

   clReleaseMemObject(m_deviceDataPointer);
}



/*!
 *  Copies data from device memory to another device memory.
 *
 *  \param copyToPointer The destination address.
 *  \param numElements Number of elements to copy, default value -1 = all elements.
 *  \param dstOffset Offset (if any) in destination pointer.
 *  \param srcOffset Offset (if any) in source pointer.
 */
template <typename T>
void DeviceMemPointer_CL<T>::copyDeviceToDevice(cl_mem copyToPointer,int numElements, int dstOffset, int srcOffset) const
{
   if(m_hostDataPointer != NULL)
   {
      DEBUG_TEXT_LEVEL1("DEVICE_TO_DEVICE!!!\n")

      cl_int err;
      size_t sizeVec;

      if(numElements == -1)
         sizeVec = m_numElements*sizeof(T);
      else
         sizeVec = numElements*sizeof(T);

#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
      clFinish(m_device->getQueue());
#endif
      copyUpTimer.start();
#endif

      err = clEnqueueCopyBuffer(m_device->getQueue(),m_deviceDataPointer, copyToPointer, srcOffset*sizeof(T), dstOffset*sizeof(T), sizeVec, 0, NULL, NULL);

      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error copying data to device. size: " << sizeVec << "\n");
      }

#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
      clFinish(m_device->getQueue());
#endif
      copyUpTimer.stop();
#endif

      deviceDataHasChanged = true;
   }
}


/*!
 *  Copies data from host memory to device memory. An internal method not used publcially.
 *  It allows copying with offset
 *
 *  \param src_ptr The source address.
 *  \param dst_ptr The destination address.
 *  \param numElements Number of elements to copy, default value -1 = all elements.
 *  \param offset The offset in the device buffer.
 */
template <typename T>
void DeviceMemPointer_CL<T>::copyHostToDevice_internal(T* src_ptr, cl_mem dest_ptr, int numElements, int offset) const
{
   DEBUG_TEXT_LEVEL1("HOST_TO_DEVICE INTERNAL!!!\n")

   cl_int err;
   size_t sizeVec;

   sizeVec = numElements*sizeof(T);

#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
   clFinish(m_device->getQueue());
#endif
   copyUpTimer.start();
#endif

   err = clEnqueueWriteBuffer(m_device->getQueue(), dest_ptr, CL_TRUE, offset*sizeof(T), sizeVec, (void*)src_ptr, 0, NULL, NULL);

   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error copying data to device. size: "<< sizeVec << "\n");
   }

#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
   clFinish(m_device->getQueue());
#endif
   copyUpTimer.stop();
#endif

   if(!m_initialized) // set that it is initialized
      m_initialized = true;

//    deviceDataHasChanged = false;
}


/*!
 *  Copies data from host memory to device memory.
 *
 *  \param numElements Number of elements to copy, default value -1 = all elements.
 *  \param copyLast Boolean flag specifying whether should copy last updated copy only (default: false).
 */
template <typename T>
void DeviceMemPointer_CL<T>::copyHostToDevice(int numElements, bool copyLast) const
{
   DEBUG_TEXT_LEVEL1("HOST_TO_DEVICE!!!\n")

   cl_int err;
   size_t sizeVec;
   if(numElements == -1)
      if(copyLast)
         sizeVec = m_numElements*sizeof(T);
      else
         sizeVec = m_effectiveNumElements*sizeof(T);
   else
      sizeVec = numElements*sizeof(T);

#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
   clFinish(m_device->getQueue());
#endif
   copyUpTimer.start();
#endif

   if(copyLast)
      err = clEnqueueWriteBuffer(m_device->getQueue(), m_deviceDataPointer, CL_TRUE, 0, sizeVec, (void*)m_hostDataPointer, 0, NULL, NULL);
   else
      err = clEnqueueWriteBuffer(m_device->getQueue(), m_effectiveDeviceDataPointer, CL_TRUE, 0, sizeVec, (void*)m_effectiveHostDataPointer, 0, NULL, NULL);

   if(err != CL_SUCCESS)
   {
      SKEPU_ERROR("Error copying data to device. size: " << sizeVec << "\n");
   }

#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
   clFinish(m_device->getQueue());
#endif
   copyUpTimer.stop();
#endif

   if(!m_initialized) // set that it is initialized
      m_initialized = true;

   deviceDataHasChanged = false;
}

/*!
 *  Copies data from device memory to host memory. Only copies if data on device has been marked as changed.
 *
 *  \param numElements Number of elements to copy, default value -1 = all elements.
 *  \param copyLast Boolean flag specifying whether should copy last updated copy only (default: false).
 */
template <typename T>
void DeviceMemPointer_CL<T>::copyDeviceToHost(int numElements, bool copyLast) const
{
   if(deviceDataHasChanged)
   {
      DEBUG_TEXT_LEVEL1("DEVICE_TO_HOST!!!\n")

      cl_int err;
      size_t sizeVec;
      if(numElements == -1)
         if(copyLast)
            sizeVec = m_numElements*sizeof(T);
         else
            sizeVec = m_effectiveNumElements*sizeof(T);
      else
         sizeVec = numElements*sizeof(T);

#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
      clFinish(m_device->getQueue());
#endif
      copyDownTimer.start();
#endif

      if(copyLast)
         err = clEnqueueReadBuffer(m_device->getQueue(), m_deviceDataPointer, CL_TRUE, 0, sizeVec, (void*)m_hostDataPointer, 0, NULL, NULL);
      else
         err = clEnqueueReadBuffer(m_device->getQueue(), m_effectiveDeviceDataPointer, CL_TRUE, 0, sizeVec, (void*)m_effectiveHostDataPointer, 0, NULL, NULL);

      if(err != CL_SUCCESS)
      {
         SKEPU_ERROR("Error copying data from device. size:"<<sizeVec<<"\n");
      }

#ifdef SKEPU_MEASURE_TIME_DISTRIBUTION
#ifdef SKEPU_MEASURE_ONLY_COPY
      clFinish(m_device->getQueue());
#endif
      copyDownTimer.stop();
#endif
      deviceDataHasChanged = false;
   }
}

/*!
 *  \return OpenCL memory object representing data on the device.
 */
template <typename T>
cl_mem DeviceMemPointer_CL<T>::getDeviceDataPointer() const
{
   return m_deviceDataPointer;
}

/*!
 *  Marks the device data as changed.
 */
template <typename T>
void DeviceMemPointer_CL<T>::changeDeviceData()
{
   DEBUG_TEXT_LEVEL1("CHANGE_DEVICE_DATA!!!\n")
   deviceDataHasChanged = true;
   m_initialized = true;
}

}

#endif

#endif

