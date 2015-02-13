namespace skepu
{

/*!
 *  \class Vector::iterator
 *  \author Johan Enmyren, Usman Dastgeer
 *  \version 0.7
 *
 *  \brief An vector iterator class.
 *
 *  An iterator class for \p skepu::Vector. behaves like the vector iterator for \p std::vector
 *  but similar to \p skepu::Vector it sometimes returns a \p proxy_elem instead of the actual
 *  element. Also makes sure the vector is properly synchronized with device before returning
 *  any elements.
 */
template <typename T>
class Vector<T>::iterator
{

public: //-- Constructors & Destructor --//

   iterator(Vector<T>& vec, T *std_iterator);

public: //-- Types --//

#ifdef SKEPU_CUDA
   typedef typename Vector<T>::device_pointer_type_cu device_pointer_type_cu;
#endif

#ifdef SKEPU_OPENCL
   typedef typename Vector<T>::device_pointer_type_cl device_pointer_type_cl;
#endif

   typedef typename Vector<T>::value_type value_type;
   typedef typename Vector<T>::size_type size_type;
   typedef typename Vector<T>::difference_type difference_type;
   typedef typename Vector<T>::pointer pointer;
   typedef typename Vector<T>::reference reference;
   typedef Vector<T> parent_type;

public: //-- Extras --//

   Vector<T>& getParent() const;
   T* getAddress() const;

   //Does not care about device data, use with care...sometimes pass negative indices...
   T& operator()(const ssize_t index = 0);
   
   //Does not care about device data, use with care...sometimes pass negative indices...
   const T& operator()(const ssize_t index) const;

public: //-- Operators --//

   T& operator[](const ssize_t index);
   const T& operator[](const ssize_t index) const;

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

   iterator operator-(const ssize_t i) const;
   iterator operator+(const ssize_t i) const;

   typename Vector<T>::difference_type operator-(const iterator& i) const;
//    typename skepu::Vector<T>::iterator::difference_type skepu::Vector<T>::iterator::operator-(const skepu::Vector<T>::iterator&) const

   T& operator *();
   const T& operator* () const;

   const T& operator-> () const;
   T& operator-> ();


private: //-- Data --//

   Vector<T>& m_parent;

   T *m_std_iterator;
};

template <typename T>
Vector<T>::iterator::iterator(Vector<T>& parent, T *std_iterator) : m_parent(parent), m_std_iterator(std_iterator) {}


template <typename T>
Vector<T>& Vector<T>::iterator::getParent() const
{
   return m_parent;
}

template <typename T>
T* Vector<T>::iterator::getAddress() const
{
   return m_std_iterator;
}

template <typename T>
T& Vector<T>::iterator::operator()(const ssize_t index)
{
   return m_std_iterator[index];
}

template <typename T>
const T& Vector<T>::iterator::operator()(const ssize_t index) const 
{
   return m_std_iterator[index];
}

template <typename T>
T& Vector<T>::iterator::operator[](const ssize_t index)
{
   m_parent.updateHost();
   m_parent.invalidateDeviceData();

   return m_std_iterator[index];
}

template <typename T>
const T& Vector<T>::iterator::operator[](const ssize_t index) const
{
   m_parent.updateHost();

   return m_std_iterator[index];
}



template <typename T>
bool Vector<T>::iterator::operator==(const iterator& i)
{
   return (m_std_iterator == i.m_std_iterator);
}

template <typename T>
bool Vector<T>::iterator::operator!=(const iterator& i)
{
   return (m_std_iterator != i.m_std_iterator);
}

template <typename T>
bool Vector<T>::iterator::operator<(const iterator& i)
{
   return (m_std_iterator < i.m_std_iterator);
}

template <typename T>
bool Vector<T>::iterator::operator>(const iterator& i)
{
   return (m_std_iterator > i.m_std_iterator);
}

template <typename T>
bool Vector<T>::iterator::operator<=(const iterator& i)
{
   return (m_std_iterator <= i.m_std_iterator);
}

template <typename T>
bool Vector<T>::iterator::operator>=(const iterator& i)
{
   return (m_std_iterator >= i.m_std_iterator);
}

template <typename T>
const typename Vector<T>::iterator& Vector<T>::iterator::operator++() //Prefix
{
   ++m_std_iterator;
   return *this;
}

template <typename T>
typename Vector<T>::iterator Vector<T>::iterator::operator++(int) //Postfix
{
   iterator temp(*this);
   ++m_std_iterator;
   return temp;
}

template <typename T>
const typename Vector<T>::iterator& Vector<T>::iterator::operator--() //Prefix
{
   --m_std_iterator;
   return *this;
}

template <typename T>
typename Vector<T>::iterator Vector<T>::iterator::operator--(int) //Postfix
{
   iterator temp(*this);
   --m_std_iterator;
   return temp;
}

template <typename T>
const typename Vector<T>::iterator& Vector<T>::iterator::operator+=(const ssize_t i)
{
   m_std_iterator += i;
   return *this;
}

template <typename T>
const typename Vector<T>::iterator& Vector<T>::iterator::operator-=(const ssize_t i)
{
   m_std_iterator -= i;
   return *this;
}

template <typename T>
typename Vector<T>::iterator Vector<T>::iterator::operator-(const ssize_t i) const
{
   iterator temp(*this);
   temp -= i;
   return temp;
}

template <typename T>
typename Vector<T>::iterator Vector<T>::iterator::operator+(const ssize_t i) const
{
   iterator temp(*this);
   temp += i;
   return temp;
}

template <typename T>
typename Vector<T>::difference_type Vector<T>::iterator::operator-(const iterator& i) const
{
   return m_std_iterator - i.m_std_iterator;
}

template <typename T>
T& Vector<T>::iterator::operator*()
{
   m_parent.updateHost();
   m_parent.invalidateDeviceData();

   return *m_std_iterator;
}

template <typename T>
const T& Vector<T>::iterator::operator*() const
{
   m_parent.updateHost();

   return *m_std_iterator;
}

template <typename T>
const T& Vector<T>::iterator::operator-> () const
{
   m_parent.updateHost();

   return *m_std_iterator;
}

template <typename T>
T& Vector<T>::iterator::operator-> ()
{
   m_parent.updateHost();
   m_parent.invalidateDeviceData();

   return *m_std_iterator;
}

}

