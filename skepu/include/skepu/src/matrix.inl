/*! \file matrix.inl
 *  \brief Contains the definitions of non-backend specific member functions for the Matrix container.
 */

namespace skepu
{


/*!
* Private helper to swap any two elements of same type
*/
template<typename T>
template<typename Type>
void Matrix<T>::item_swap(Type &t1, Type &t2)
{
   Type temp= t1;
   t1=t2;
   t2=temp;
}


///////////////////////////////////////////////
// Operators START
///////////////////////////////////////////////




/*!
 *  copy matrix,,, copy row and column count as well along with data
 */
template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other)
{
   if(*this == other)
      return *this;
   
   other.updateHost();
   invalidateDeviceData();

   m_data = other.m_data;
   m_rows = other.m_rows;
   m_cols = other.m_cols;
   return *this;
}


/*!
 *  resize matrix,,, invalidates all copies before resizing.
 */
template <typename T>
void Matrix<T>::resize(size_type _rows, size_type _cols, T val)
{
   if (_rows == m_rows && _cols == m_cols)
   {
      return;
   }

   updateHostAndInvalidateDevice();

   typename Matrix<T>::container_type m( _rows*_cols, val);
   typename Matrix<T>::size_type colSize = std::min(m_cols,_cols) * sizeof(T);
   typename Matrix<T>::size_type minRow = std::min(m_rows,_rows);

   for (size_type r=0; r < _rows; r++)
   {
      for(size_type c=0; c < _cols; c++)
      {
         if(r < m_rows && c < m_cols)
            m[(r * _cols + c)]= m_data[(r * m_cols + c)];
         else
            m[(r * _cols + c)]= val;
      }
   }

   m_data=m;
   m_rows = _rows;
   m_cols = _cols;

}

/*!
 *  Add \p rhs matrix operation element wise to current matrix. Two matrices must be of same size.
 * \param rhs The matrix which is used in addition to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator+=(const Matrix<T>& rhs)
{
   if(m_rows != rhs.m_rows || m_cols != rhs.m_cols)
      SKEPU_ERROR("ERROR: Matrix should be of same size for this operation!");

   rhs.updateHost();
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] += rhs.m_data[r*m_cols+c];
      }
   return *this;
}


/*!
 *  Adds a scalar value to all elements in the current matrix.
 * \param rhs The value which is used in addition to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator+=(const T& rhs)
{
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] += rhs;
      }
   return *this;
}

/*!
 *  Subtract \p rhs matrix operation element wise to current matrix. Two matrices must be of same size.
 * \param rhs The matrix which is used in subtraction to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator-=(const Matrix<T>& rhs)
{
   rhs.updateHost();
   updateHostAndInvalidateDevice();

   if(m_rows != rhs.m_rows || m_cols != rhs.m_cols)
      SKEPU_ERROR("ERROR: Matrix should be of same size for this operation!");

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] -= rhs.m_data[r*m_cols+c];
      }
   return *this;
}

/*!
 *  Subtracts a scalar value to all elements in the current matrix.
 * \param rhs The value which is used in subtraction to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator-=(const T& rhs)
{
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] -= rhs;
      }
   return *this;
}


/*!
 *  Multiplies \p rhs matrix operation element wise to current matrix. Two matrices must be of same size. NB it is not matrix multiplication
 * \param rhs The matrix which is used in multiplication to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator*=(const Matrix<T>& rhs)
{
   rhs.updateHost();
   updateHostAndInvalidateDevice();

   if(m_rows != rhs.m_rows || m_cols != rhs.m_cols)
      SKEPU_ERROR("ERROR: Matrix should be of same size for this operation!");

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] *= rhs.m_data[r*m_cols+c];
      }
   return *this;
}

/*!
 *  Multiplies a scalar value to all elements in the current matrix.
 * \param rhs The value which is used in multiplication to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator*=(const T& rhs)
{
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] *= rhs;
      }
   return *this;
}

/*!
 *  Divides \p rhs matrix operation element wise to current matrix. Two matrices must be of same size. NB it is not matrix multiplication
 * \param rhs The matrix which is used in division to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator/=(const Matrix<T>& rhs)
{
   rhs.updateHost();
   updateHostAndInvalidateDevice();

   if(m_rows != rhs.m_rows || m_cols != rhs.m_cols)
      SKEPU_ERROR("ERROR: Matrix should be of same size for this operation!");

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] /= rhs.m_data[r*m_cols+c];
      }
   return *this;
}

/*!
 *  Divides a scalar value to all elements in the current matrix.
 * \param rhs The value which is used in division to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator/=(const T& rhs)
{
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] /= rhs;
      }
   return *this;
}


/*!
 *  Taking Mod with \p rhs matrix, element wise to current matrix. Two matrices must be of same size.
 * \param rhs The value which is used in taking mod to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator%=(const Matrix<T>& rhs)
{
   rhs.updateHost();
   updateHostAndInvalidateDevice();
   if(m_rows != rhs.m_rows || m_cols != rhs.m_cols)
      SKEPU_ERROR("ERROR: Matrix should be of same size for this operation!");

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] %= rhs.m_data[r*m_cols+c];
      }
   return *this;
}

/*!
 *  Taking Mod with a scalar value to all elements in the current matrix.
 * \param rhs The value which is used in taking mod to current matrix.
 */
template <typename T>
const Matrix<T>& Matrix<T>::operator%=(const T& rhs)
{
   updateHostAndInvalidateDevice();

   for(size_type r=0; r<m_rows; r++)
      for(size_type c=0; c<m_cols; c++)
      {
         m_data[r*m_cols+c] %= rhs;
      }
   return *this;
}


///////////////////////////////////////////////
// Operators END
///////////////////////////////////////////////

///////////////////////////////////////////////
// Public Helpers START
///////////////////////////////////////////////

/*!
    *  Updates the matrix from its device allocations.
    */
template <typename T>
inline void Matrix<T>::updateHost() const
{
#ifdef SKEPU_OPENCL
   updateHost_CL();
#endif

#ifdef SKEPU_CUDA
   /*! the m_valid logic is only implemented for CUDA backend. The OpenCL still uses the old memory management mechanism */
   if(m_valid) // if already up to date then no need to check...
      return;
   
   updateHost_CU();
   
   m_valid = true;
#endif
}

/*!
 *  Invalidates (mark copies data invalid) all device data that this matrix has allocated.
 */
template <typename T>
inline void Matrix<T>::invalidateDeviceData()
{
   /// this flag is used to track whether contents in main matrix are changed so that the contents of the 
   /// transpose matrix that was taken earlier need to be updated again...
   /// normally invalidation occurs when contents are changed so good place to update this flag (?)
   m_dataChanged = true; 
   
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
 *  First updates the matrix from its device allocations. Then invalidates (mark copies data invalid) the data allocated on devices.
 */
template <typename T>
inline void Matrix<T>::updateHostAndInvalidateDevice()
{
   updateHost();
   invalidateDeviceData();
}

/*!
 *  Removes the data copies allocated on devices.
 */
template <typename T>
inline void Matrix<T>::releaseDeviceAllocations()
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
 *  First updates the matrix from its device allocations. Then removes the data copies allocated on devices.
 */
template <typename T>
inline void Matrix<T>::updateHostAndReleaseDeviceAllocations()
{
   updateHost();
   releaseDeviceAllocations();
}




///////////////////////////////////////////////
// Regular interface functions START
///////////////////////////////////////////////

//Iterators
/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::begin()
{
   return iterator(this, m_data.begin());
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator. Uses \p row to get an iterator for that row.
 * \param row The index of row from where to start iterator.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::begin(unsigned row)
{
   if(row>=total_rows())
   {
      std::cerr<<"ERROR! Row index is out of bound!\n";
      throw;
   }
   return iterator(this, m_data.begin()+(row*total_cols()));
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator.
 */
template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::begin() const
{
   return m_data.begin();
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator. Uses \p row to get an iterator for that row.
 * \param row The index of row from where to start iterator.
 */
template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::begin(unsigned row) const
{
   if(row>=total_rows())
   {
      std::cerr<<"ERROR! Row index is out of bound!\n";
      throw;
   }
   return iterator(this, m_data.begin()+(row*total_cols()));
}


/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::end()
{
   return iterator(this, m_data.end());
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator. Get iterator to last element of \p row.
 * \param row Index of row the iterator will point to the last element.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::end(unsigned row)
{
   if(row>=total_rows())
   {
      std::cerr<<"ERROR! Row index is out of bound!\n";
      throw;
   }
   return iterator(this, m_data.end()-((total_rows()-(row+1))*total_cols()));
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator.
 */
template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::end() const
{
   return m_data.end();
}

/*!
 *  Please refer to the documentation of \p std::vector and \p skepu::Matrix::iterator. Get iterator to last element of \p row.
 * \param row Index of row the iterator will point to the last element.
 */
template <typename T>
typename Matrix<T>::const_iterator Matrix<T>::end(unsigned row) const
{
   if(row>=total_rows())
   {
      std::cerr<<"ERROR! Row index is out of bound!\n";
      throw;
   }
   return iterator(this, m_data.end()-((total_rows()-(row+1))*total_cols()));
}


/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
typename Matrix<T>::size_type Matrix<T>::capacity() const
{
   return m_data.capacity();
}



/*!
 *  Please refer to the documentation of \p std::vector.
 */
template <typename T>
bool Matrix<T>::empty() const
{
   return m_data.empty();
}




/*!
 *  Please refer to the documentation of \p std::vector.
 *
 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
 *  behaves like an ordinary, but there might be exceptions.
 */
template <typename T>
typename Matrix<T>::proxy_elem Matrix<T>::at(size_type row, size_type col)
{
   return proxy_elem(*this, row*m_cols+col);
}

/*!
 *  To initialize a matrix with soem scalar value.
 *
 *  \param elem The element you want to assign to all matrix.
 */
template <typename T>
Matrix<T>& Matrix<T>::operator=(const T& elem)
{
   for(size_type i=0; i<size(); i++)
   {
      m_data[i]= elem;
   }
   return *this;
}


/*!
 *  Please refer to the documentation of \p std::vector. Uses \p row and \p col instead of single index.
 *  \param row Index of row to get.
 *  \param col Index of column to get.
 *  \return a const reference to T element at position identified by row,column index.
 */
template <typename T>
const T& Matrix<T>::at(size_type row, size_type col ) const
{
   updateHost();
   if(row >= this->total_rows() || col >= this->total_cols())
      throw "ERROR! Row or Column index is out of bound!";

   return m_data.at(row*m_cols+col);
}

/*!
 *  To get a subsection of matrix. This will creat a separate copy.
 *  \param row Index of row to get.
 * \param rowWidth Width of the row of new Matrix.
 *  \param col Index of column to get.
 * \param colWidth Width of column of new Matrix.
 */
template <typename T>
Matrix<T>& Matrix<T>::subsection(size_type row, size_type col, size_type rowWidth, size_type colWidth)
{
   updateHost();

   if(row+rowWidth>= total_rows())
      throw "ERROR! row index and width is larger than total rows!\n";

   if(col+colWidth>= total_cols())
      throw "ERROR! column index and column width is larger than total columns!\n";


   Matrix<T> *submat=new Matrix<T>(rowWidth, colWidth);
   std::cout<<submat->total_rows()<<"   "<<submat->total_cols();
   for(typename Matrix<T>::size_type r=row, rsub=0; rsub<rowWidth; r++, rsub++)
   {
      for(typename Matrix<T>::size_type c=col, csub=0; csub<colWidth; c++, csub++)
      {
         submat->at(rsub,csub)= this->at(r,c);
      }
   }
   return *submat;
}

/*!
 *  Return index of last element of \p row.
 *
 * \param row Index of the row.
 */
template <typename T>
typename Matrix<T>::size_type Matrix<T>::row_back(size_type row)
{
   if(row>=m_rows)
      throw "Row index out of bound exception";

   updateHost();
   typename Matrix<T>::size_type index= ((row+1) * m_cols)-1;
   return index;
}



/*!
 *  Return last element of \p row.
 *
 * \param row Index of the row.
 */
template <typename T>
const T& Matrix<T>::row_back(size_type row) const
{
   if(row>=m_rows)
      throw "Row index out of bound exception";
   updateHost();
   typename Matrix<T>::size_type index= ((row+1) * m_cols)-1;
   return m_data[index];
}

/*!
 *  Return index of first element of \p row in 1D container.
 *
 * \param row Index of the row.
 */
template <typename T>
typename Matrix<T>::size_type Matrix<T>::row_front(size_type row)
{
   if(row>=m_rows)
      throw "Row index out of bound exception";
   updateHost();
   return (row * m_cols);
}

/*!
 *  Return first element of \p row.
 *
 * \param row Index of the row.
 */
template <typename T>
const T& Matrix<T>::row_front(size_type row) const
{
   if(row>=m_rows)
      SKEPU_ERROR("Row index out of bound exception");

   updateHost();
   return m_data[(row * m_cols)];
}


/*!
 *  Returns proxy of last element in \p column.
 *
 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
 *  behaves like an ordinary, but there might be exceptions.
 * \p col Index of the column.
 */
template <typename T>
typename Matrix<T>::proxy_elem Matrix<T>::col_back(size_type col)
{
   if(col>=m_cols)
      throw "Column index out of bound exception";

   typename Matrix<T>::size_type index= ((m_rows-1)*m_cols)+col;
   return proxy_elem(*this, index);
}

/*!
 *  Returns last element in \p column.
 *
 * \p col Index of the column.
 */
template <typename T>
const T& Matrix<T>::col_back(size_type col) const
{

   if(col>=m_cols)
      throw "Column index out of bound exception";

   updateHost();
   typename Matrix<T>::size_type index= ((m_rows-1)*m_cols)+col;
   return m_data[index];
}


/*!
 *  Returns proxy of first element in \p column.
 *
 *  Returns a \p proxy_elem instead of an ordinary element. The \p proxy_elem usually
 *  behaves like an ordinary, but there might be exceptions.
 * \p col Index of the column.
 */
template <typename T>
typename Matrix<T>::proxy_elem Matrix<T>::col_front(size_type col)
{
   if(col>=m_cols)
      throw "Column index out of bound exception";
   return proxy_elem(*this, col);
}

/*!
 *  Returns last element in \p column.
 *
 * \p col Index of the column.
 */
template <typename T>
const T& Matrix<T>::col_front(size_type col) const
{

   if(col>=m_cols)
      throw "Column index out of bound exception";
   updateHost();
   return m_data[col];
}


/*!
 *  Please refer to the documentation of \p std::vector.
 * Invalidates all copies before clear.
 */
template <typename T>
void Matrix<T>::clear()
{
   invalidateDeviceData();

   m_data.clear();
}

/*!
 *  Please refer to the documentation of \p std::vector.
 * Updates and invalidate both Matrices before swapping.
 */
template <typename T>
void Matrix<T>::swap(Matrix<T>& from)
{
   updateHostAndInvalidateDevice();
   from.updateHostAndInvalidateDevice();

   item_swap<typename Matrix<T>::size_type>(m_rows, from.m_rows);
   item_swap<typename Matrix<T>::size_type>(m_cols, from.m_cols);
   item_swap<typename Matrix::container_type>(m_data, from.m_data);
}


/*!
 *  Please refer to the documentation of \p std::vector.
 * Updates and invalidate the Matrix.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::erase( typename Matrix<T>::iterator loc )
{
   updateHostAndInvalidateDevice();

   return iterator(m_data.erase(loc), *this);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 * Erases a certain number of elements pointed by \p start and \p end. Updates and Invalidates all copies before.
 */
template <typename T>
typename Matrix<T>::iterator Matrix<T>::erase( typename Matrix<T>::iterator start, typename Matrix<T>::iterator end )
{
   updateHostAndInvalidateDevice();

   return iterator(m_data.erase(start, end), *this);
}



///////////////////////////////////////////////
// Regular interface functions END
///////////////////////////////////////////////


///////////////////////////////////////////////
// Additions to interface START
///////////////////////////////////////////////


/*!
 *  Flushes the matrix, synchronizing it with the device then release all device allocations.
 */
template <typename T>
void Matrix<T>::flush()
{
#ifdef SKEPU_OPENCL
   flush_CL();
#endif

#ifdef SKEPU_CUDA
   flush_CU();
#endif
}


/*!
 *  Behaves like \p operator[] and unlike \p skepu::Vector, it cares about synchronizing with device.
 *  Can be used when accessing to access elements row and column wise.
 *
 *  \param row Index to a specific row of the Matrix.
 *  \param col Index to a specific column of the Matrix.
 */
template <typename T>
const T& Matrix<T>::operator()(const size_type row, const size_type col) const
{
   updateHost();
   if(row >= this->total_rows() || col >= this->total_cols())
      throw "ERROR! Row or Column index is out of bound!";
   return m_data[row * m_cols + col];
}


/*!
 *  Behaves like \p operator[] and unlike \p skepu::Vector, it cares about synchronizing with device.
 *  Can be used when accessing to access elements row and column wise.
 *
 *  \param row Index to a specific row of the Matrix.
 *  \param col Index to a specific column of the Matrix.
 */
template <typename T>
T& Matrix<T>::operator()(const size_type row, const size_type col)
{
   updateHostAndInvalidateDevice();
   if(row >= this->total_rows() || col >= this->total_cols())
      throw "ERROR! Row or Column index is out of bound!";
   return m_data[row * m_cols + col];
}

/*!
 *  Behaves like \p operator[] but does not care about synchronizing with device.
 *  Can be used when accessing many elements quickly so that no synchronization
 *  overhead effects performance. Make sure to properly synch with device by calling
 *  updateHost etc before use.
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
T& Matrix<T>::operator()(const size_type index)
{
   return m_data[index];
}

/*!
 *  A \p operator[] that care about synchronizing with device.
 *  Can be used when accessing elements considering consecutive storage
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
const T& Matrix<T>::operator[](const size_type index) const
{
   updateHost();
   if(index >= (this->total_rows() * this->total_cols()))
      throw "ERROR! Index is out of bound!";
   return m_data[index];
}

/*!
 *  A \p operator[] that care about synchronizing with device.
 *  Can be used when accessing elements considering consecutive storage
 *
 *  \param index Index of element assuming continuous Matrix row-wise storage. To facilitate access using single indexing
 */
template <typename T>
T& Matrix<T>::operator[](const size_type index)
{
   updateHostAndInvalidateDevice();
   if(index >= (this->total_rows() * this->total_cols()))
      throw "ERROR! Index is out of bound!";
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
 *
 */
template <typename T>
bool Matrix<T>::operator==(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data == m_data);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator!=(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data != m_data);
}


/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator<(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data < m_data);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator>(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data > m_data);
}

/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator<=(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data <= m_data);
}


/*!
 *  Please refer to the documentation of \p std::vector.
 *
 */
template <typename T>
bool Matrix<T>::operator>=(const Matrix<T>& c1)
{
   c1.updateHost();
   updateHost();

   return (c1.m_data >= m_data);
}


} // end namespace skepu

