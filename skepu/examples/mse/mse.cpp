/*!
* MSE (mean Square Error)  The MSE represents the cumulative squared error between the compressed and the original image.
* Which is practically the "sum" of square differences between the original and compressed images, divided by image size.
*/

// following define to enable/disable OpenMP implmentation to be used
/* #define SKEPU_OPENMP */

// following define to enable/disable OpenCL implmentation to be used
/* #define SKEPU_OPENCL */

#include <iostream>

#include "skepu/vector.h"
#include "skepu/map.h"
#include "skepu/reduce.h"
#include <math.h>


BINARY_FUNC(diff_f, float, a, b, return ((a-b)*(a-b));)

BINARY_FUNC(sum_f, float, a, b, return a+b;)

#define ROWS 16
#define COLS 12

int main()
{
   skepu::Map<diff_f> map_diff(new diff_f);

   skepu::Reduce<sum_f> red_sum(new sum_f);

   skepu::Vector<float> in_act_img(ROWS*COLS);
   skepu::Vector<float> in_comp_img(ROWS*COLS);
   skepu::Vector<float> out_img(ROWS*COLS);

   in_act_img.randomize(1,255); // set to some random values in this case, can be initializaed from an actual image
   in_comp_img.randomize(1,255); // set to some random values in this case, can be initializaed from an actual image

   map_diff(in_act_img, in_comp_img, out_img);

   float mse = (red_sum(out_img) / ROWS*COLS);

   std::cout<<"MSE of two images: "<< mse <<"\n";

   return 0;
}

