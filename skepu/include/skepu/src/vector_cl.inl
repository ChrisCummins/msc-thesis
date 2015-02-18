#ifdef SKEPU_OPENCL

namespace skepu
{

/*!
 *  \brief Update device with vector content.
 *
 *  Update device with a vector range. If vector does not have an allocation on the device for
 *  the current range, create a new allocation and if specified, also copy vector data to device.
 *  Saves newly allocated ranges to \p m_deviceMemPointers_CL so vector can keep track of where
 *  and what it has stored on devices.
 *
 *  \param start Pointer to first element in range to be updated with device.
 *  \param numElements Number of elemets in range.
 *  \param device Pointer to the device that should be synched with.
 *  \param copy Boolean value that tells whether to only allocate or also copy vector data to device. True copies, False only allocates.
 */
template <typename T>
typename Vector<T>::device_pointer_type_cl Vector<T>::updateDevice_CL(T* start, size_type numElements, Device_CL* device, bool copy)
{
   T* root= &m_data[0];
   typename std::map<std::pair< cl_device_id, T* >, device_pointer_type_cl >::iterator result;
   std::pair< cl_device_id, T* > key(device->getDeviceID(), root);

   result = m_deviceMemPointers_CL.find(key);

   if(result == m_deviceMemPointers_CL.end()) //insert new, alloc mem and copy
   {
      device_pointer_type_cl temp = new DeviceMemPointer_CL<T>(start, numElements, device);
      if(copy)
      {
         //Make sure uptodate
         updateHost_CL();
         //Copy
         temp->copyHostToDevice();
      }
      result = m_deviceMemPointers_CL.insert(m_deviceMemPointers_CL.begin(), std::make_pair(key,temp));
   }
   //already exists, update from host if needed
   else if(copy && !result->second->m_initialized) // we check for case when space is allocated but data was not copied, Multi-GPU case
   {
      //Make sure uptodate
      updateHost_CL(); // FIXT IT: Only check for this copy and not for all copies.
      //Copy
      result->second->copyHostToDevice();	 // internally it will set "result->second->m_initialized = true; "
   }
//    else //already exists, update from host if needed
//    {
//        //Do nothing for now, since writes to host deallocs device mem
//    }

   return result->second;
}


template <typename T>
bool Vector<T>::isVectorOnDevice_CL(Device_CL* device, bool multi)
{
   if(!multi)
   {
      typename std::map<std::pair< cl_device_id, std::pair< T*, size_type > >, device_pointer_type_cl >::iterator result;
      std::pair< cl_device_id, std::pair< T*, size_type > > key(device->getDeviceID(), std::pair< T*, size_type >(&m_data[0], size()));
      result = m_deviceMemPointers_CL.find(key);
      return !(result == m_deviceMemPointers_CL.end());
   }
   return false; // handle multi case here...
}


/*!
 *  \brief Flushes the vector.
 *
 *  First it updates the vector from all its device allocations, then it releases all allocations.
 */
template <typename T>
void Vector<T>::flush_CL()
{
   updateHost_CL();
   releaseDeviceAllocations_CL();
}

/*!
 *  \brief Updates the host from devices.
 *
 *  Updates the vector from all its device allocations.
 */
template <typename T>
inline void Vector<T>::updateHost_CL() const
{
   if(!m_deviceMemPointers_CL.empty())
   {
      typename std::map<std::pair< cl_device_id, T* >, device_pointer_type_cl >::const_iterator it;
      for(it = m_deviceMemPointers_CL.begin(); it != m_deviceMemPointers_CL.end(); ++it)
      {
         it->second->copyDeviceToHost();
      }
   }
}

/*!
 *  \brief Invalidates the device data.
 *
 *  Invalidates the device data by releasing all allocations. This way the vector is updated
 *  and then data must be copied back to devices if used again.
 */
template <typename T>
inline void Vector<T>::invalidateDeviceData_CL()
{
   //deallocs all device mem for vector for now
   if(!m_deviceMemPointers_CL.empty())
   {
      releaseDeviceAllocations_CL();
   }
   //Could maybe be made better by only setting a flag that data is not valid
}

/*!
 *  \brief Releases device allocations.
 *
 *  Releases all device allocations for this vector. The memory pointers are removed.
 */
template <typename T>
inline void Vector<T>::releaseDeviceAllocations_CL()
{
   typename std::map<std::pair< cl_device_id, T* >, device_pointer_type_cl >::const_iterator it;
   for(it = m_deviceMemPointers_CL.begin(); it != m_deviceMemPointers_CL.end(); ++it)
   {
      delete it->second;
   }
   m_deviceMemPointers_CL.clear();
}

}

#endif

