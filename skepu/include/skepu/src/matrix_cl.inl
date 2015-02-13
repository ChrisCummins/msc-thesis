/*! \file matrix_cl.inl
 *  \brief Contains the definitions of OpenCL specific member functions of the Matrix class.
 */


#ifdef SKEPU_OPENCL

namespace skepu
{

/*!
 *  \brief Update device with matrix content.
 *
 *  Update device with a Matrix range by specifying rowsize and column size. This allows to create rowwise paritions.
 *  If Matrix does not have an allocation on the device for
 *  the current range, create a new allocation and if specified, also copy Matrix data to device.
 *  Saves newly allocated ranges to \p m_deviceMemPointers_CL so matrix can keep track of where
 *  and what it has stored on devices.
 *
 *  \param start Pointer to first element in range to be updated with device.
 *  \param rows Number of rows.
 *  \param cols Number of columns.
 *  \param device Pointer to the device that should be synched with.
 *  \param copy Boolean value that tells whether to only allocate or also copy matrix data to device. True copies, False only allocates.
 */
template <typename T>
typename Matrix<T>::device_pointer_type_cl Matrix<T>::updateDevice_CL(T* start, size_type rows, size_type cols, Device_CL* device, bool copy)
{
   DEBUG_TEXT_LEVEL3("Matrix updating device OpenCL\n")

   typename std::map<std::pair< cl_device_id, std::pair< T*, size_type > >, device_pointer_type_cl >::iterator result;

   std::pair< cl_device_id, std::pair< T*, size_type > > key(device->getDeviceID(), std::pair< T*, size_type >(start, rows * cols));
   result = m_deviceMemPointers_CL.find(key);

   if(result == m_deviceMemPointers_CL.end()) //insert new, alloc mem and copy
   {
      device_pointer_type_cl temp = new DeviceMemPointer_CL<T>(start, rows * cols, device);
      if(copy)
      {
         //Make sure uptodate
         updateHost_CL();
         //Copy
         temp->copyHostToDevice();
      }
      result = m_deviceMemPointers_CL.insert(m_deviceMemPointers_CL.begin(), std::make_pair(key,temp));
   }
   else if(copy && !result->second->m_initialized) // we check for case when space is allocated but data was not copied, Multi-GPU case
   {
      //Make sure uptodate
      updateHost_CL(); // FIX IT: Only check for this copy and not for all copies.
      //Copy
      result->second->copyHostToDevice();	 // internally it will set "result->second->m_initialized = true; "
   }
//    else //already exists, update from host if needed
//    {
//        //Do nothing for now, since writes to host deallocs device mem
//    }

   return result->second;
}



/*!
 *  \brief Update device with matrix content.
 *
 *  Update device with a Matrix range by specifying rowsize only as number of rows is assumed to be 1 in this case.
 *  Helper function, useful for scenarios where matrix need to be treated like Vector 1D.
 *  If Matrix does not have an allocation on the device for
 *  the current range, create a new allocation and if specified, also copy Matrix data to device.
 *  Saves newly allocated ranges to \p m_deviceMemPointers_CL so matrix can keep track of where
 *  and what it has stored on devices.
 *
 *  \param start Pointer to first element in range to be updated with device.
 *  \param cols Number of columns.
 *  \param device Pointer to the device that should be synched with.
 *  \param copy Boolean value that tells whether to only allocate or also copy matrix data to device. True copies, False only allocates.
 */
template <typename T>
typename Matrix<T>::device_pointer_type_cl Matrix<T>::updateDevice_CL(T* start, size_type cols, Device_CL* device, bool copy)
{
   return updateDevice_CL(start, (size_type)1, cols, device, copy);
}

/*!
 *  \brief Flushes the matrix.
 *
 *  First it updates the matrix from all its device allocations, then it releases all allocations.
 */
template <typename T>
void Matrix<T>::flush_CL()
{
   DEBUG_TEXT_LEVEL3("Matrix flush OpenCL\n")

   updateHost_CL();
   releaseDeviceAllocations_CL();
}

/*!
 *  \brief Updates the host from devices.
 *
 *  Updates the matrix from all its device allocations.
 */
template <typename T>
inline void Matrix<T>::updateHost_CL() const
{
   DEBUG_TEXT_LEVEL3("Matrix updating host OpenCL\n")

   if(!m_deviceMemPointers_CL.empty())
   {
      typename std::map<std::pair< cl_device_id, std::pair< T*, size_type > >, device_pointer_type_cl >::const_iterator it;
      for(it = m_deviceMemPointers_CL.begin(); it != m_deviceMemPointers_CL.end(); ++it)
      {
         it->second->copyDeviceToHost();
      }
   }
}

/*!
 *  \brief Invalidates the device data.
 *
 *  Invalidates the device data by releasing all allocations. This way the matrix is updated
 *  and then data must be copied back to devices if used again.
 */
template <typename T>
inline void Matrix<T>::invalidateDeviceData_CL()
{
   DEBUG_TEXT_LEVEL3("Matrix invalidating device data OpenCL\n")

   //deallocs all device mem for matrix for now
   if(!m_deviceMemPointers_CL.empty())
   {
      releaseDeviceAllocations_CL();
   }
   //Could maybe be made better by only setting a flag that data is not valid
}

/*!
 *  \brief Releases device allocations.
 *
 *  Releases all device allocations for this matrix. The memory pointers are removed.
 */
template <typename T>
inline void Matrix<T>::releaseDeviceAllocations_CL()
{
   DEBUG_TEXT_LEVEL3("Matrix releasing device allocations OpenCL\n")

   typename std::map<std::pair< cl_device_id, std::pair< T*, size_type > >, device_pointer_type_cl >::const_iterator it;
   for(it = m_deviceMemPointers_CL.begin(); it != m_deviceMemPointers_CL.end(); ++it)
   {
      delete it->second;
   }
   m_deviceMemPointers_CL.clear();
}

}

#endif

