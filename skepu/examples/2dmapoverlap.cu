// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

// following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>

#include "skepu/matrix.h"
#include "skepu/mapoverlap.h"

/*!
 * user-function. For 2D-convolution we use "OVERLAP_FUNC_2D_STR"
 * Parameters are:
 *  name,
 *  datatype,
 *  overlap length on horizontal axis,
 *  overlap length on vertical axis,
 *  name of parameter,
 *  the stride which is used to access items column-wise,
 *  actual function body.
 */
OVERLAP_FUNC_2D_STR(over_f, int, 1, 2, a, stride,
                    return (a[-2*stride-1] + a[-1*stride+1] + a[-1*stride] + a[-1] + a[0] + a[1] + a[1*stride+1] + a[2*stride] + a[2*stride+1]);
                   )

// some size typedefs....
#define OUT_ROWS 16
#define OUT_COLS 10

#define OVERLAP_ROWS 2 // vertical axis
#define OVERLAP_COLS 1 // horizontal axis

#define IN_ROWS (OUT_ROWS + OVERLAP_ROWS*2)
#define IN_COLS (OUT_COLS + OVERLAP_COLS*2)

#define OUT_SIZE (OUT_ROWS * OUT_COLS)
#define IN_SIZE (IN_ROWS * IN_COLS)

int main()
{
   skepu::MapOverlap2D<over_f> mat_conv(new over_f);

   skepu::Matrix<int> m0(IN_ROWS,IN_COLS, 0);
   skepu::Matrix<int> m1(OUT_ROWS,OUT_COLS, 0);

   // initializing non-edge elements of input matrix
   for(int i=OVERLAP_ROWS; i<OUT_ROWS+OVERLAP_ROWS;	i++)
   {
      for(int j=OVERLAP_COLS; j<OUT_COLS+OVERLAP_COLS;	j++)
      {
         m0(i,j)=i+j;
      }
   }

   std::cout<<"Input "<<m0<<"\n";

   // Applying 2D convolution for neghbouring elements
   mat_conv(m0,m1);

   std::cout<<"Output "<<m1<<"\n";

   return 0;
}