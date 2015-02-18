/*! \file deviceprop_cl.h
 *  \brief Declares a struct used to store OpenCL device properties.
 */

#ifndef DEVICEPROP_CL_H
#define DEVICEPROP_CL_H

#ifdef SKEPU_OPENCL

#ifdef USE_MAC_OPENCL
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <vector>

namespace skepu
{

/*!
 *  \ingroup helpers
 */

/*!
 *  \struct openclGenProp
 *
 *  A helper struct to openclDeviceProp. Used to help with the fetching of properties. See Device_CL.
 */
struct openclGenProp
{
   cl_device_info param_name;
   size_t param_value_size;
   void* param_value;
};

/*!
 *  \ingroup helpers
 */

/*!
 *  \struct openclDeviceProp
 *
 *  A struct used to store OpenCL device properties. Adds the neccessary properties to a list which
 *  can be used to fetch all those properties in a for-loop. See Device_CL.
 */
struct openclDeviceProp
{
   openclDeviceProp()
   {
      openclGenProp temp;

      temp.param_name = CL_DEVICE_ADDRESS_BITS;
      temp.param_value_size = sizeof(cl_uint);
      temp.param_value = (void*)&DEVICE_ADDRESS_BITS;
      propertyList.push_back(temp);

      temp.param_name = CL_DEVICE_MAX_WORK_GROUP_SIZE;
      temp.param_value_size = sizeof(size_t);
      temp.param_value = (void*)&DEVICE_MAX_WORK_GROUP_SIZE;
      propertyList.push_back(temp);

      temp.param_name = CL_DEVICE_MAX_COMPUTE_UNITS;
      temp.param_value_size = sizeof(cl_uint);
      temp.param_value = (void*)&DEVICE_MAX_COMPUTE_UNITS;
      propertyList.push_back(temp);

      temp.param_name = CL_DEVICE_GLOBAL_MEM_SIZE;
      temp.param_value_size = sizeof(cl_ulong);
      temp.param_value = (void*)&DEVICE_GLOBAL_MEM_SIZE;
      propertyList.push_back(temp);

      temp.param_name = CL_DEVICE_LOCAL_MEM_SIZE;
      temp.param_value_size = sizeof(cl_ulong);
      temp.param_value = (void*)&DEVICE_LOCAL_MEM_SIZE;
      propertyList.push_back(temp);
   }

   std::vector<openclGenProp> propertyList;

   cl_uint DEVICE_ADDRESS_BITS;
   cl_bool DEVICE_AVAILABLE;
   cl_bool DEVICE_COMPILER_AVAILABLE;
   cl_device_fp_config DEVICE_DOUBLE_FP_CONFIG;
   cl_bool DEVICE_ENDIAN_LITTLE;
   cl_bool DEVICE_ERROR_CORRECTION_SUPPORT;
   cl_device_exec_capabilities DEVICE_EXECUTION_CAPABILITIES;
   char* DEVICE_EXTENSIONS;
   cl_ulong DEVICE_GLOBAL_MEM_CACHE_SIZE;
   cl_device_mem_cache_type DEVICE_GLOBAL_MEM_CACHE_TYPE;
   cl_uint DEVICE_GLOBAL_MEM_CACHELINE_SIZE;
   cl_ulong DEVICE_GLOBAL_MEM_SIZE;
   cl_device_fp_config DEVICE_HALF_FP_CONFIG;
   cl_bool DEVICE_IMAGE_SUPPORT;
   size_t DEVICE_IMAGE2D_MAX_HEIGHT;
   size_t DEVICE_IMAGE2D_MAX_WIDTH;
   size_t DEVICE_IMAGE3D_MAX_DEPTH;
   size_t DEVICE_IMAGE3D_MAX_HEIGHT;
   size_t DEVICE_IMAGE3D_MAX_WIDTH;
   cl_ulong DEVICE_LOCAL_MEM_SIZE;
   cl_device_local_mem_type DEVICE_LOCAL_MEM_TYPE;
   cl_uint DEVICE_MAX_CLOCK_FREQUENCY;
   cl_uint DEVICE_MAX_COMPUTE_UNITS;
   cl_uint DEVICE_MAX_CONSTANT_ARGS;
   cl_ulong DEVICE_MAX_CONSTANT_BUFFER_SIZE;
   cl_ulong DEVICE_MAX_MEM_ALLOC_SIZE;
   size_t DEVICE_MAX_PARAMETER_SIZE;
   cl_uint DEVICE_MAX_READ_IMAGE_ARGS;
   cl_uint DEVICE_MAX_SAMPLERS;
   size_t DEVICE_MAX_WORK_GROUP_SIZE;
   cl_uint DEVICE_MAX_WORK_ITEM_DIMENSIONS;
   size_t DEVICE_MAX_WORK_ITEM_SIZES[3];
   cl_uint DEVICE_MAX_WRITE_IMAGE_ARGS;
   cl_uint DEVICE_MEM_BASE_ADDR_ALIGN;
   cl_uint DEVICE_MIN_DATA_TYPE_ALIGN_SIZE;
   char* DEVICE_NAME;
   cl_platform_id DEVICE_PLATFORM;
   cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_CHAR;
   cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_SHORT;
   cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_INT;
   cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_LONG;
   cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT;
   cl_uint DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE;
   char* DEVICE_PROFILE;
   size_t DEVICE_PROFILING_TIMER_RESOLUTION;
   cl_command_queue_properties DEVICE_QUEUE_PROPERTIES;
   cl_device_fp_config DEVICE_SINGLE_FP_CONFIG;
   cl_device_type DEVICE_TYPE;
   char* DEVICE_VENDOR;
   cl_uint DEVICE_VENDOR_ID;
   char* DEVICE_VERSION;
   char* DRIVER_VERSION;
};

}

#endif

#endif

