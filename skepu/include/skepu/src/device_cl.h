/*! \file device_cl.h
 *  \brief Contains a class declaration for the object that represents an OpenCL device.
 */

#ifndef DEVICE_CL_H
#define DEVICE_CL_H

#ifdef SKEPU_OPENCL

#include <iostream>
#ifdef USE_MAC_OPENCL
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "deviceprop_cl.h"


namespace skepu
{

/*!
 *  \ingroup helpers
 */

/*!
 *  \class Device_CL
 *
 *  \brief A class representing an OpenCL device.
 *
 *  This class represents one OpenCL device which can be used to execute the skeletons on if OpenCL
 *  is used as backend. Stores various properties about the device and provides functions that return them.
 *  Also contains a OpenCL context and queue.
 */
class Device_CL
{

private:

   cl_device_type m_type;
   openclDeviceProp m_deviceProp;

   cl_device_id m_device;
   cl_context m_context;
   cl_command_queue m_queue;

   size_t m_maxThreads;
   size_t m_maxBlocks;

   /*!
    *  Run once during construction to get all the device properties.
    *
    *  \param device OpenCL device ID.
    */
   void getDeviceProps(cl_device_id device)
   {
      for(std::vector<openclGenProp>::iterator it = m_deviceProp.propertyList.begin(); it != m_deviceProp.propertyList.end(); ++it)
      {
         cl_int err;
         err = clGetDeviceInfo(device, it->param_name, it->param_value_size, it->param_value, NULL);
         if(err != CL_SUCCESS)
         {
            std::cerr<<"Error adding property value CL!!\n";
         }
      }
   }


public:

   /*!
    *  The constructor creates a device from an ID, device type (should be GPU in this version)
    *  and a context. It gets all the properties and creates a command-queue.
    *
    *  \param id Device ID for the device that is to be created.
    *  \param type The OpenCL device type.
    *  \param context A valid OpenCL context.
    */
   Device_CL(cl_device_id id, cl_device_type type, cl_context context)
   {
      m_device = id;
      m_type = type;
      m_context = context;

      getDeviceProps(id);

      m_maxThreads = getMaxBlockSize()>>1;
      m_maxBlocks = (size_t)((size_t)1<<(m_deviceProp.DEVICE_ADDRESS_BITS-1))*2-1;

      cl_int err;

      //Create a command-queue on the GPU device
      m_queue = clCreateCommandQueue(m_context, m_device, 0, &err);
      if(err != CL_SUCCESS)
      {
         std::cerr<<"Error creating queue!!\n" <<err <<"\n";
      }
   }

   /*!
    *  The destructor releases the OpenCL queue and context.
    */
   ~Device_CL()
   {
      clReleaseCommandQueue(m_queue);
      clReleaseContext(m_context);
      std::cout<<"Release Device_cl\n";
   }

   /*!
    *  \return The maximum block (work group) size.
    */
   size_t getMaxBlockSize() const
   {
      return m_deviceProp.DEVICE_MAX_WORK_GROUP_SIZE;
   }

   /*!
    *  \return The maximum number of compute units available.
    */
   cl_uint getNumComputeUnits() const
   {
      return m_deviceProp.DEVICE_MAX_COMPUTE_UNITS;
   }

   /*!
    *  \return The global memory size.
    */
   cl_ulong getGlobalMemSize() const
   {
      return m_deviceProp.DEVICE_GLOBAL_MEM_SIZE;
   }

   /*!
    *  \return The local (shared) memory size.
    */
   cl_ulong getSharedMemPerBlock() const
   {
      return m_deviceProp.DEVICE_LOCAL_MEM_SIZE;
   }

   /*!
    *  \return The maximum number of threads per block or group.
    */
   int getMaxThreads() const
   {
#ifdef SKEPU_MAX_GPU_THREADS
      return SKEPU_MAX_GPU_THREADS;
#else
      return m_maxThreads;
#endif
   }

   /*!
    *  \return The maximum number of blocks or groups for a kernel launch.
    */
   size_t getMaxBlocks() const
   {
#ifdef SKEPU_MAX_GPU_BLOCKS
      return SKEPU_MAX_GPU_BLOCKS;
#else
      return m_maxBlocks;
#endif
   }

   /*!
    *  \return OpenCL context.
    */
   const cl_context& getContext() const
   {
      return m_context;
   }

   /*!
    *  \return OpenCL queue.
    */
   const cl_command_queue& getQueue() const
   {
      return m_queue;
   }

   /*!
    *  \return OpenCL device type.
    */
   cl_device_type getType() const
   {
      return m_type;
   }

   /*!
    *  \return OpenCL device ID.
    */
   cl_device_id getDeviceID() const
   {
      return m_device;
   }

};


}

#endif

#endif

