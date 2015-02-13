/*! \file 2dmapoverlap_omp.inl
 *  \brief Contains the definitions of OpenCL specific member functions for the MapOverlap2D skeleton.
 */
#ifdef SKEPU_OPENMP

#include <iostream>
#include <omp.h>


#include "operator_type.h"

namespace skepu
{

/*!
 *  Performs the 2D MapOverlap on a Matrix on the \em OpenMP with the same Matrix as output.
 *  The actual filter is specified in a user-function.
 *
 *  \param input A Matrix that is used for both input and output.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::OMP(Matrix<T>& input)
{
   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t filter_rows=m_mapOverlapFunc->overlapY;
   size_t filter_cols=m_mapOverlapFunc->overlapX;

   size_t out_rows=in_rows-(filter_rows*2);
   size_t out_cols=in_cols-(filter_cols*2);

   Matrix<T> output(out_rows,out_cols);

   OMP(input, output);

   size_t k=0;
   for(size_t i= filter_rows; i<(out_rows+filter_rows); i++)
      for(size_t j=filter_cols; j<(out_cols+filter_cols); j++)
      {
         input(i*in_cols+j) = output(k++);
      }
}



/*!
 *  Performs the 2D MapOverlap on a whole matrix on the \em OpenMP with a separate output matrix.
 *  The actual filter is specified in a user-function.
 *
 *  \param input A matrix which the mapping will be performed on.
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::OMP(Matrix<T>& input, Matrix<T>& output)
{
   DEBUG_TEXT_LEVEL1("2D MAPOVERLAP OpenMP\n")

   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t out_rows=output.total_rows();
   size_t out_cols=output.total_cols();

   size_t filter_rows=m_mapOverlapFunc->overlapY;
   size_t filter_cols=m_mapOverlapFunc->overlapX;

   if( ( (in_rows-(filter_rows*2)) != out_rows) && ( (in_cols-(filter_cols*2)) != out_cols))
   {
      output.clear();
      output.resize(in_rows-(filter_rows*2), in_cols-(filter_cols*2));
   }


   m_mapOverlapFunc->setStride(in_cols);

   #pragma omp parallel for shared(input, output, in_rows, in_cols, out_rows, out_cols, filter_rows, filter_cols) num_threads(omp_get_max_threads())
   for(size_t y=0; y<out_rows; y++)
      for(size_t x=0; x<out_cols; x++)
      {
         output(y*out_cols+x) = m_mapOverlapFunc->CPU(&input((y+filter_rows)*in_cols + (x+filter_cols))); //sum / (filter_rows * filter_cols);
      }
}





/*!
 *  Performs the 2D MapOverlap on the \em OpenMP, based on provided filter and input neighbouring elements on a whole Matrix
 *  With a separate Matrix as output.
 *
 *  \param input A matrix which the mapping will be performed on. It should include padded data as well considering the filter size
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param filter The filter which will be applied for each element in the output.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::OMP(Matrix<T>& input, Matrix<T>& output, Matrix<T>& filter)
{
   DEBUG_TEXT_LEVEL1("2D MAPOVERLAP with Filter Matrix OpenMP\n")
   
   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t out_rows=output.total_rows();
   size_t out_cols=output.total_cols();

   size_t filter_rows=filter.total_rows();
   size_t filter_cols=filter.total_cols();

   if( ( (in_rows-filter_rows+1) != out_rows) && ( (in_cols-filter_cols+1) != out_cols))
   {
      output.clear();
      output.resize((in_rows-filter_rows+1), (in_cols-filter_cols+1));
   }

   T sum;

   #pragma omp parallel for default(shared) private(sum)
   for(size_t y=0; y<out_rows; y++)
      for(size_t x=0; x<out_cols; x++)
      {
         sum=0;
         for(size_t j=0; j<filter_rows; j++)
         {
            for(size_t i=0; i<filter_cols; i++)
            {
               sum += input((y+j) , (x+i)) * filter(j,i);
            }
         }
         output(y,x) = sum / (filter_rows * filter_cols);
      }
}



/*!
 *  Performs the 2D MapOverlap on the \em OpenMP, by taking average of neighbouring elements on a whole Matrix.
 *  With a separate Matrix as output.
 *
 *  \param input A matrix which the mapping will be performed on. It should include padded data as well considering the filter size
 *  \param output The result matrix, will be overwritten with the result and resized if needed.
 *  \param filter_rows The number of rows used as neighbouring elements to calculate new value for each output element.
 *  \param filter_cols The number of columns used as neighbouring elements to calculate new value for each output element.
 */
template <typename MapOverlap2DFunc>
template <typename T>
void MapOverlap2D<MapOverlap2DFunc>::OMP(Matrix<T>& input, Matrix<T>& output, size_t filter_rows, size_t filter_cols)
{
   DEBUG_TEXT_LEVEL1("2D MAPOVERLAP with Average Filter OpenMP\n")
   
   //change filter_rows filter_cols to a different representation, used internally in this implementation
   filter_rows = filter_rows*2+1;
   filter_cols = filter_cols*2+1;

   size_t in_rows=input.total_rows();
   size_t in_cols=input.total_cols();

   size_t out_rows=output.total_rows();
   size_t out_cols=output.total_cols();

   if( ( (in_rows-filter_rows+1) != out_rows) && ( (in_cols-filter_cols+1) != out_cols))
   {
      output.clear();
      output.resize((in_rows-filter_rows+1), (in_cols-filter_cols+1));
   }

   T sum;

   #pragma omp parallel for default(shared) private(sum)
   for(size_t y=0; y<out_rows; y++)
      for(size_t x=0; x<out_cols; x++)
      {
         sum=0;
         for(size_t j=0; j<filter_rows; j++)
         {
            for(size_t i=0; i<filter_cols; i++)
            {
               sum += input((y+j) , (x+i));
            }
         }
         output(y,x) = sum / (filter_rows * filter_cols);
      }
}


}

#endif

