/*! \file matrix_iterator.inl
 *  \brief Contains the definitions of Matrix::iterator class.
 */

namespace skepu
{

/*!
 *  \class Matrix::iterator
 *
 *  \brief An matrix iterator class that tranverses row-wise.
 *
 *  An iterator class for \p skepu::Matrix. It traverses a Matrix row-wise assuming Matrix is stored in row-major order
 *  which is \p skepu::Matrix default style. It behaves like the 1D container iterators like iterator for \p std::vector
 *  but similar to \p skepu::Matrix it sometimes returns a \p proxy_elem instead of the actual
 *  element. Also makes sure the matrix is properly synchronized with device before returning
 *  any elements.
 */
template <typename T>
class Matrix<T>::iterator
{

public: //-- Constructors & Destructor --//
#ifdef USE_PINNED_MEMORY
   iterator(Matrix<T> *mat, const typename std::vector<T, malloc_allocator<T> >::iterator std_iterator);
#else
   iterator(Matrix<T> *mat, const typename std::vector<T>::iterator std_iterator);
#endif


public: //-- Types --//

#ifdef SKEPU_CUDA
   typedef typename Matrix<T>::device_pointer_type_cu device_pointer_type_cu;
#endif

#ifdef SKEPU_OPENCL
   typedef typename Matrix<T>::device_pointer_type_cl device_pointer_type_cl;
#endif

   typedef typename Matrix<T>::value_type value_type;
   typedef typename Matrix<T>::size_type size_type;
   typedef typename Matrix<T>::difference_type difference_type;
   typedef typename Matrix<T>::pointer pointer;
   typedef typename Matrix<T>::reference reference;

   typedef Matrix<T> parent_type;


public: //-- Extras --//

   Matrix<T>& getParent() const;
   T* getAddress() const;

   //Does care about device data, uses updateAndInvalidateDevice for read and write access
   T& operator()(const ssize_t rows, const ssize_t cols);

   //Does care about device data, uses updateDevice, for readonly access
   const T& operator()(const ssize_t rows, const ssize_t cols) const;

   //Does not care about device data, use with care
   T& operator()(const ssize_t index=0);

public: //-- Operators --//

   T& operator[](const ssize_t index);

   const T& operator[](const ssize_t index) const;


   operator const_iterator() const;
   operator typename std::vector<T>::iterator() const;

   bool operator==(const iterator& i);
   bool operator!=(const iterator& i);
   bool operator<(const iterator& i);
   bool operator>(const iterator& i);
   bool operator<=(const iterator& i);
   bool operator>=(const iterator& i);

   const iterator& operator++();
   iterator operator++(int);
   const iterator& operator--();
   iterator operator--(int);

   const iterator& operator+=(const ssize_t i);
   const iterator& operator-=(const ssize_t i);

   iterator& stride_row(const ssize_t stride=1);


   iterator operator-(const ssize_t i) const;
   iterator operator+(const ssize_t i) const;

   typename Matrix<T>::difference_type operator-(const iterator& i) const;

   T& operator *();
   const T& operator* () const;

   const T& operator-> () const;
   T& operator-> ();


private: //-- Data --//

   Matrix<T>* m_parent;

#ifdef USE_PINNED_MEMORY
   typename std::vector<T, malloc_allocator<T> >::iterator m_std_iterator;
#else
   typename std::vector<T>::iterator m_std_iterator;
#endif


};

#ifdef USE_PINNED_MEMORY

template <typename T>
Matrix<T>::iterator::iterator(Matrix<T> *parent, const typename std::vector<T, malloc_allocator<T> >::iterator std_iterator) : m_parent(parent), m_std_iterator(std_iterator) {}

#else

template <typename T>
Matrix<T>::iterator::iterator(Matrix<T> *parent, const typename std::vector<T>::iterator std_iterator) : m_parent(parent), m_std_iterator(std_iterator) {}

#endif

template <typename T>
Matrix<T>& Matrix<T>::iterator::getParent() const
{
   return (*m_parent);
}

template <typename T>
T* Matrix<T>::iterator::getAddress() const
{
   return &(*m_std_iterator);
}

// Does not care about device data, use with care...
template <typename T>
T& Matrix<T>::iterator::operator()(const ssize_t index)
{
   return m_std_iterator[index];
}

template <typename T>
T& Matrix<T>::iterator::operator()(const ssize_t row, const ssize_t col)
{
   m_parent->updateHost();
   m_parent->invalidateDeviceData();

   return m_std_iterator[(row*getParent().total_cols() + col)];
}


template <typename T>
const T& Matrix<T>::iterator::operator()(const ssize_t row, const ssize_t col) const
{
   m_parent->updateHost();

   return m_std_iterator[(row*getParent().total_cols() + col)];
}

template <typename T>
typename Matrix<T>::iterator& Matrix<T>::iterator::stride_row(const ssize_t stride)
{
   return m_std_iterator += (stride * getParent().total_cols());
}



template <typename T>
T& Matrix<T>::iterator::operator[](const ssize_t index)
{
   m_parent->updateHost();
   m_parent->invalidateDeviceData();

   return m_std_iterator[index];
}




template <typename T>
const T& Matrix<T>::iterator::operator[](const ssize_t index) const
{
   m_parent->updateHost();

   return m_std_iterator[index];
}




template <typename T>
Matrix<T>::iterator::operator const_iterator() const
{
   m_parent->updateHost();

   return static_cast< const_iterator > (m_std_iterator);
}


template <typename T>
Matrix<T>::iterator::operator typename std::vector<T>::iterator() const
{
   m_parent->updateHost();
   m_parent->invalidateDeviceData();

   return m_std_iterator;
}

template <typename T>
bool Matrix<T>::iterator::operator==(const iterator& i)
{
   return (m_std_iterator == i.m_std_iterator);
}

template <typename T>
bool Matrix<T>::iterator::operator!=(const iterator& i)
{
   return (m_std_iterator != i.m_std_iterator);
}

template <typename T>
bool Matrix<T>::iterator::operator<(const iterator& i)
{
   return (m_std_iterator < i.m_std_iterator);
}

template <typename T>
bool Matrix<T>::iterator::operator>(const iterator& i)
{
   return (m_std_iterator > i.m_std_iterator);
}

template <typename T>
bool Matrix<T>::iterator::operator<=(const iterator& i)
{
   return (m_std_iterator <= i.m_std_iterator);
}

template <typename T>
bool Matrix<T>::iterator::operator>=(const iterator& i)
{
   return (m_std_iterator >= i.m_std_iterator);
}

template <typename T>
const typename Matrix<T>::iterator& Matrix<T>::iterator::operator++() //Prefix
{
   ++m_std_iterator;
   return *this;
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::iterator::operator++(int) //Postfix
{
   iterator temp(*this);
   ++m_std_iterator;
   return temp;
}

template <typename T>
const typename Matrix<T>::iterator& Matrix<T>::iterator::operator--() //Prefix
{
   --m_std_iterator;
   return *this;
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::iterator::operator--(int) //Postfix
{
   iterator temp(*this);
   --m_std_iterator;
   return temp;
}

template <typename T>
const typename Matrix<T>::iterator& Matrix<T>::iterator::operator+=(const ssize_t i)
{
   m_std_iterator += i;
   return *this;
}

template <typename T>
const typename Matrix<T>::iterator& Matrix<T>::iterator::operator-=(const ssize_t i)
{
   m_std_iterator -= i;
   return *this;
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::iterator::operator-(const ssize_t i) const
{
   iterator temp(*this);
   temp -= i;
   return temp;
}

template <typename T>
typename Matrix<T>::iterator Matrix<T>::iterator::operator+(const ssize_t i) const
{
   iterator temp(*this);
   temp += i;
   return temp;
//return *this;
}

template <typename T>
typename Matrix<T>::difference_type Matrix<T>::iterator::operator-(const iterator& i) const
{
   return m_std_iterator - i.m_std_iterator;
}

template <typename T>
T& Matrix<T>::iterator::operator*()
{
   m_parent->updateHost();
   m_parent->invalidateDeviceData();
   
   return *m_std_iterator;
}

template <typename T>
const T& Matrix<T>::iterator::operator*() const
{
   m_parent->updateHost();

   return *m_std_iterator;
}

template <typename T>
const T& Matrix<T>::iterator::operator-> () const
{
   m_parent->updateHost();

   return *m_std_iterator;
}

template <typename T>
T& Matrix<T>::iterator::operator-> ()
{
   m_parent->updateHost();
   m_parent->invalidateDeviceData();

   return *m_std_iterator;
}

}

