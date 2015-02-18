/*! \file sparse_matrix_iterator.inl
 *  \brief Contains the definitions of the SparseMatrix::iterator class.
 */

namespace skepu
{

/*!
 *  \class SparseMatrix::iterator
 *
 *  \brief An sparse matrix iterator class that tranverses row-wise.
 *
 *  An iterator class for \p skepu::SparseMatrix. It traverses a SparseMatrix elements range
 *  which is \p skepu::Matrix default style. It behaves like the 1D container iterators like iterator for \p std::vector
 *  but similar to \p skepu::Matrix it sometimes returns a \p proxy_elem instead of the actual
 *  element. Also makes sure the matrix is properly synchronized with device before returning
 *  any elements.
 */
template <typename T>
class SparseMatrix<T>::iterator
{

public: //-- Constructors & Destructor --//
   iterator(SparseMatrix<T> *parent, T *start, T *end);


public: //-- Extras --//

   SparseMatrix<T>& getParent() const;
   T* getAddress() const;

   //Does not care about device data, use with care
   T& operator()(const ssize_t index);

   //Does care about device data, uses updateDevice, for readonly access
   const T& operator()(const ssize_t index) const;

public: //-- Operators --//

   //Does care about device data, uses updateDevice, for readwrite access
   T& operator[](const ssize_t index);

   //Does care about device data, uses updateDevice, for readonly access
   const T& operator[](const ssize_t index) const;

   size_t size()
   {
      return m_size;
   }


private: //-- Data --//
   size_t m_size;
   SparseMatrix<T>* m_parent;
   T *m_start;
   T *m_end;
};


template <typename T>
SparseMatrix<T>::iterator::iterator(SparseMatrix<T>* parent, T *start, T *end) : m_parent(parent), m_start(start), m_end(end)
{
   SKEPU_ASSERT(m_start);
   SKEPU_ASSERT(m_end);
   SKEPU_ASSERT((m_end-m_start)>=0);
   m_size = (m_end-m_start);
}

template <typename T>
T* SparseMatrix<T>::iterator::getAddress() const
{
   return m_start;
}

// Does not care about device data, use with care...
template <typename T>
const T& SparseMatrix<T>::iterator::operator()(const ssize_t index) const
{
   return m_start[index];
}

template <typename T>
T& SparseMatrix<T>::iterator::operator()(const ssize_t index)
{
   return m_start[index];
}


template <typename T>
const T& SparseMatrix<T>::iterator::operator[](const ssize_t index) const
{
#ifdef SKEPU_OPENCL
   m_parent->updateHost_CL();
#endif

#ifdef SKEPU_CUDA
   m_parent->updateHost_CU();
#endif

   return m_start[index];
}


template <typename T>
T& SparseMatrix<T>::iterator::operator[](const ssize_t index)
{
#ifdef SKEPU_OPENCL
   m_parent->updateHost_CL();
   m_parent->invalidateDeviceData_CL();
#endif

#ifdef SKEPU_CUDA
   m_parent->updateHost_CU();
   m_parent->invalidateDeviceData_CU();
#endif

   return m_start[index];
}


}

