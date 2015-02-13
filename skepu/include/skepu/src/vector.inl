namespace skepu
{

///////////////////////////////////////////////
// Constructors START
///////////////////////////////////////////////


/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
inline Vector<T>::Vector(): m_capacity(10), m_size(0), m_deallocEnabled(true), m_valid(true), m_noValidDeviceCopy(true)
{
   allocateHostMemory<T>(m_data, m_capacity);
}

/*!
 *  Please refer to the documentation of \p std::vector. The copy occurs w.r.t. elements.
 *  As copy constructor creates a new storage.
 *
 *  Updates vector \p c before copying.
 */
template <typename T>
inline Vector<T>::Vector(const Vector& c): m_capacity(c.m_capacity), m_size(c.m_size), m_deallocEnabled(true), m_valid(true), m_noValidDeviceCopy(true)
{
   if(m_size<1)
      throw std::out_of_range("The vector size should be positive.\n");

   allocateHostMemory<T>(m_data, m_capacity);

   c.updateHost();

   std::copy(c.m_data, c.m_data + m_size, m_data);
}


/**!
 * Used to construct vector on a raw data pointer passed to it as its payload data.
 * Useful when creating the vector object with existing raw data pointer.
 */
template <typename T>
inline Vector<T>::Vector(T * const ptr, size_type size, bool deallocEnabled): m_capacity(size), m_size (size), m_deallocEnabled(deallocEnabled), m_valid(true), m_noValidDeviceCopy(true)
{
   if(m_size<1)
      throw std::out_of_range("The vector size should be positive.\n");

   if(!ptr)
   {
      std::cerr<<"Error: The supplied pointer for initializing vector object is invalid\n";
      return;
   }

   m_data = ptr;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
inline Vector<T>::Vector(size_type num, const T& val): m_capacity(num), m_size(num), m_deallocEnabled(true), m_valid(true), m_noValidDeviceCopy(true)
{
   if(m_size<1)
      throw std::out_of_range("The vector size should be positive.\n");

//    m_data = new T[m_capacity];
   allocateHostMemory<T>(m_data, m_capacity);

   std::fill(m_data, m_data + m_size, val);
   //    for(size_type i=0;i<m_size;i++)
   //    {
   //           m_data[i]=val;
   //    }
}




///////////////////////////////////////////////
// Constructors END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Destructor START
///////////////////////////////////////////////

/*!
 *  Releases all allocations made on device.
 */
template <typename T>
Vector<T>::~Vector()
{
   releaseDeviceAllocations();

   if(m_data && m_deallocEnabled)
   {
//       std::cerr << "deleting memory: " << m_nameVerbose << "\n";
      deallocateHostMemory<T>(m_data);
   }
}

///////////////////////////////////////////////
// Destructor END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Public Helpers START
///////////////////////////////////////////////

/*!
 *  Updates the vector from its device allocations.
 */
template <typename T>
inline void Vector<T>::updateHost() const
{
#ifdef SKEPU_OPENCL
   updateHost_CL();
#endif

#ifdef SKEPU_CUDA
   /*! the m_valid logic is only implemented for CUDA backend. The OpenCL still uses the old memory management mechanism */
   if(m_valid) // if already up to date then no need to check...
      return;
   
   updateHost_CU();
#endif

   m_valid = true;
}

/*!
 *  Invalidates (mark copies data invalid) all device data that this vector has allocated.
 */
template <typename T>
inline void Vector<T>::invalidateDeviceData()
{
#ifdef SKEPU_OPENCL
   invalidateDeviceData_CL();
#endif

#ifdef SKEPU_CUDA
   if(m_noValidDeviceCopy)
       assert(m_valid);
   
   if(!m_noValidDeviceCopy)
   {
      invalidateDeviceData_CU();
      m_noValidDeviceCopy = true;
      m_valid = true;
   }
#endif
}

/*!
 *  First updates the vector from its device allocations. Then invalidates (mark copies data invalid) the data allocated on devices.
 */
template <typename T>
inline void Vector<T>::updateHostAndInvalidateDevice()
{
   updateHost();
   invalidateDeviceData();
}

/*!
 *  Removes the data copies allocated on devices.
 */
template <typename T>
inline void Vector<T>::releaseDeviceAllocations()
{
#ifdef SKEPU_OPENCL
   releaseDeviceAllocations_CL();
#endif

#ifdef SKEPU_CUDA
   m_valid = true;
   
   releaseDeviceAllocations_CU();
#endif
}

/*!
 *  First updates the vector from its device allocations. Then removes the data copies allocated on devices.
 */
template <typename T>
inline void Vector<T>::updateHostAndReleaseDeviceAllocations()
{
   updateHost();
   releaseDeviceAllocations();
}





///////////////////////////////////////////////
// Public Helpers END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Operators START
///////////////////////////////////////////////

/*!
 *  Please refer to the documentation of \p std::vector.
 *
 *  Returns a proxy_elem instead of an ordinary element. The proxy_elem usually
 *  behaves like an ordinary, but there might be exceptions.
 */
template <typename T>
typename Vector<T>::proxy_elem Vector<T>::operator[](const size_type index)
{
   return proxy_elem(*this, index);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
const T& Vector<T>::operator[](const size_type index) const
{
   updateHost();
//    updateHostAndInvalidateDevice();

   return m_data[index];
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
Vector<T>& Vector<T>::operator=(const Vector<T>& other)
{
   if(*this == other)
      return *this;
   
   updateHostAndReleaseDeviceAllocations();
   other.updateHost();

   if(m_capacity<other.m_size)
   {
      if(m_data)
      {
//          std::cerr << "deleting memory: " << m_nameVerbose << "\n";
         deallocateHostMemory<T>(m_data);
      }

      m_capacity = m_size = other.m_size;

      allocateHostMemory<T>(m_data, m_capacity);
   }
   else
   {
      m_size = other.m_size;
   }

   std::copy(other.m_data, other.m_data + m_size, m_data);

   return *this;
}

///////////////////////////////////////////////
// Operators END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Regular interface functions START
///////////////////////////////////////////////

//Iterators
/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
typename Vector<T>::iterator Vector<T>::begin()
{
   return iterator(*this, &m_data[0]);
}


/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
typename Vector<T>::iterator Vector<T>::end()
{
   return iterator(*this, &m_data[m_size]);
}


//Capacity
/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
typename Vector<T>::size_type Vector<T>::capacity() const
{
   return m_capacity;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
typename Vector<T>::size_type Vector<T>::size() const
{
   return m_size;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
typename Vector<T>::size_type Vector<T>::max_size() const
{
   return 1073741823;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
void Vector<T>::resize(size_type num, T val)
{
   updateHostAndReleaseDeviceAllocations();

   if(num<=m_size) // dont shrink the size, maybe good in some cases?
   {
      m_size = num;
      return;
   }

   reserve(num);

   for(size_type i=m_size; i<num; ++i)
   {
      m_data[i] = val;
   }
   m_size = num;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
bool Vector<T>::empty() const
{
   return (m_size==0);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
void Vector<T>::reserve( size_type size )
{
   if(size <= m_capacity)
      return;

   updateHostAndReleaseDeviceAllocations();
   
   T* temp;
   allocateHostMemory<T>(temp, size);
   
   std::copy(m_data, m_data + m_size, temp);
   
//    std::cerr << "deleting memory: " << m_nameVerbose << "\n";
   deallocateHostMemory<T>(m_data);

   m_data = temp;
   m_capacity = size;
   temp = 0;
}


//Element access
/*!
 *  Please refer to the documentation of \p std::vector.
 *
 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
 *  behaves like an ordinary, but there might be exceptions.
 */
template <typename T>
typename Vector<T>::proxy_elem Vector<T>::at(size_type loc)
{
   return proxy_elem(*this, loc);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
const T& Vector<T>::at( size_type loc ) const
{
   updateHost();

   return m_data[loc];
}

/*!
 *  Please refer to the documentation of \p std::vector.
 *
 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
 *  behaves like an ordinary, but there might be exceptions.
 */
template <typename T>
typename Vector<T>::proxy_elem Vector<T>::back()
{
   return proxy_elem(*this, m_size-1);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
const T& Vector<T>::back() const
{
   updateHost();

   return m_data[m_size-1];
}

/*!
 *  Please refer to the documentation of \p std::vector.
 *
 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
 *  behaves like an ordinary, but there might be exceptions.
 */
template <typename T>
typename Vector<T>::proxy_elem Vector<T>::front()
{
   return proxy_elem(*this, 0);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
const T& Vector<T>::front() const
{
   updateHost();

   return m_data[0];
}


//Modifiers
/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
void Vector<T>::assign(size_type num, const T& val)
{
   releaseDeviceAllocations();

   reserve(num); // check if need some reallocation

   std::fill(m_data, m_data+num, val);
   //   for(size_type i=0; i<num; i++)
   //   {
   //           m_data[i]=val;
   //   }

   m_size = num;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
template<typename input_iterator>
void Vector<T>::assign( input_iterator start, input_iterator end )
{
   updateHostAndReleaseDeviceAllocations();

   size_type num= end-start;

   reserve(num); // check if need some reallocation

   for(size_type i=0; i<num; i++, start++)
   {
      m_data[i]=*start;
   }

   m_size = num;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
void Vector<T>::clear()
{
   releaseDeviceAllocations();

   m_size=0;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
typename Vector<T>::iterator Vector<T>::erase( typename Vector<T>::iterator loc )
{
   updateHostAndReleaseDeviceAllocations();

   std::copy(loc+1, end(), loc);
   --m_size;
   return loc;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
typename Vector<T>::iterator Vector<T>::erase( typename Vector<T>::iterator start, typename Vector<T>::iterator end )
{
   updateHostAndReleaseDeviceAllocations();

   std::copy(end, iterator(*this, &m_data[m_size]), start);
   m_size-= (end-start);
   return start;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
typename Vector<T>::iterator Vector<T>::insert( typename Vector<T>::iterator loc, const T& val )
{
   updateHostAndReleaseDeviceAllocations();

   reserve(m_size+1);

   copy(loc, end(), loc+1);

   ++m_size;

   *loc=val;

   return loc;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
void Vector<T>::insert( typename Vector<T>::iterator loc, size_type num, const T& val )
{
   updateHostAndReleaseDeviceAllocations();

   reserve(m_size+num);

   copy(loc, end(), loc+num);
   m_size += num;

   for(size_type i=0; i<num; i++)
   {
      *loc = val;
      ++loc;
   }
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
void Vector<T>::pop_back()
{
   updateHostAndReleaseDeviceAllocations();

   --m_size;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
void Vector<T>::push_back(const T& val)
{
   updateHostAndReleaseDeviceAllocations();

   if (m_size >= m_capacity)
      reserve(m_capacity + 5);

   m_data[m_size++] = val;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
void Vector<T>::swap(Vector<T>& from)
{
   updateHostAndReleaseDeviceAllocations();
   from.updateHostAndReleaseDeviceAllocations();

   std::swap(m_data, from.m_data);
   std::swap(m_size, from.m_size);
   std::swap(m_capacity, from.m_capacity);
}

///////////////////////////////////////////////
// Regular interface functions END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Additions to interface START
///////////////////////////////////////////////

/*!
 *  Flushes the vector, synchronizing it with the device then release all device allocations.
 */
template <typename T>
void Vector<T>::flush()
{
#ifdef SKEPU_OPENCL
   flush_CL();
#endif

#ifdef SKEPU_CUDA
   flush_CU();
#endif
}

/*!
 *  Behaves like \p operator[] but does not care about synchronizing with device.
 *  Can be used when accessing many elements quickly so that no synchronization
 *  overhead effects performance. Make sure to properly synch with device by calling
 *  updateHost etc before use.
 *
 *  \param index Index to a specific element of the vector.
 */
template <typename T>
T& Vector<T>::operator()(const size_type index)
{
   return m_data[index];
}

///////////////////////////////////////////////
// Additions to interface END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Comparison operators START
///////////////////////////////////////////////

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
bool Vector<T>::operator==(const Vector<T>& c1)
{
   c1.updateHost();
   updateHost();

   if(m_size!=c1.m_size)
      return false;

   if(m_data==c1.m_data)
      return true;

   for(size_type i=0; i<m_size; i++)
   {
      if(m_data[i]!=c1.m_data[i])
         return false;
   }
   return true;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
bool Vector<T>::operator!=(const Vector<T>& c1)
{
   c1.updateHost();
   updateHost();

   if(m_size!=c1.m_size)
      return true;

   if(m_data==c1.m_data && m_size== c1.m_size)
      return false;

   for(size_type i=0; i<m_size; i++)
   {
      if(m_data[i]!=c1.m_data[i])
         return true;
   }
   return false;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
bool Vector<T>::operator<(const Vector<T>& c1)
{
   c1.updateHost();
   updateHost();

   size_type t_size = ( (c1.size()<size()) ? c1.size() : size() );

   for(size_type i=0; i<t_size; ++i)
   {
      if(m_data[i] >= c1.m_data[i])
         return false;
   }
   return true;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
bool Vector<T>::operator>(const Vector<T>& c1)
{
   c1.updateHost();
   updateHost();

   size_type t_size = ( (c1.size()<size()) ? c1.size() : size() );

   for(size_type i=0; i<t_size; ++i)
   {
      if(m_data[i] <= c1.m_data[i])
         return false;
   }
   return true;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
bool Vector<T>::operator<=(const Vector<T>& c1)
{
   c1.updateHost();
   updateHost();

   size_type t_size = ( (c1.size()<size()) ? c1.size() : size() );

   for(size_type i=0; i<t_size; ++i)
   {
      if(m_data[i] > c1.m_data[i])
         return false;
   }
   return true;
}

/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
bool Vector<T>::operator>=(const Vector<T>& c1)
{
   c1.updateHost();
   updateHost();

   size_type t_size = ( (c1.size()<size()) ? c1.size() : size() );

   for(size_type i=0; i<t_size; ++i)
   {
      if(m_data[i] < c1.m_data[i])
         return false;
   }
   return true;
}

///////////////////////////////////////////////
// Comparison operators END
///////////////////////////////////////////////

}

