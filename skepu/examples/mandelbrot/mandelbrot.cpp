/*!
*

Mandelbrot fractals. The Mandelbrot set
{
"B. B. Mandelbrot. Fractal aspects of the iteration of z → λz(1 − z) for complex λ and z.
Annals of the New York Academy of Sciences, 357:249–259, December 1980."
}
is a set of complex numbers which boundary draws a fractal in the complex numbers plane. A complex number c lies
within the Mandelbrot set, if the sequence
z_{i+1} = z_{i}2 + c
with i ∈ N, starting with z0 = 0 does not escape to infinity, otherwise c is not part of
the Mandelbrot set.

When computing a Mandelbrot fractal, the sequence in equation 3.1 is calculated
for every pixel of an image representing a section of the complex numbers plane. If a
given threshold is crossed, it is presumed that the sequence will escape to infinity and
that the pixel is not inside the Mandelbrot set. If the threshold is not crossed for a given
number of steps in the sequence, the pixel is taken as a member of the Mandelbrot
set. A pixel within the Mandelbrot set painted in black, other pixels are given a color
that corresponds to the number of sequence steps that have been calculated before
excluding the pixel from the Mandelbrot set. By setting the threshold and the number
of sequence steps accordingly, the calculation of the fractal can be a time-consuming
task. However, as all pixels are calculated independently, it is a common benchmark
application for data-parallel computations.

*/


// following define to enable/disable OpenMP implmentation to be used
/* #define SKEPU_OPENMP */

// following define to enable/disable OpenCL implmentation to be used
/* #define SKEPU_OPENCL */

#include <iostream>

#include "skepu/vector.h"
#include "skepu/map.h"
#include "skepu/reduce.h"

#include <fstream>
#include <cstdlib>



BINARY_FUNC(mandelBrote_f, float, a, b, return a*a+b;)


#define WIDTH 4096
#define HEIGHT 3072

#define CENTER_X -0.73
#define CENTER_Y -0.16
#define ZOOM 27615


#define ITER 1

void init_plane(skepu::Vector<float> &plane)
{
   float startx = CENTER_X - ((float) WIDTH / (ZOOM * 2));
   float starty = CENTER_Y - ((float) HEIGHT / (ZOOM * 2));
   float dx = (float) 1 / ZOOM;

   for (int x = 0; x < WIDTH; x++)
      for (int y = 0; y < HEIGHT; y++)
      {
         plane[x + y * WIDTH] = startx + x * dx;
      }
}

int main(int argc, char* argv[])
{
   skepu::Vector<float> in_def_img(WIDTH * HEIGHT, 0);
   skepu::Vector<float> inout_img(WIDTH * HEIGHT, 0);

   init_plane(in_def_img); /* initialize complex numbers plane */

   skepu::Map<mandelBrote_f> map_squ(new mandelBrote_f);


   for(int i=0; i<ITER; i++)
   {
      map_squ(inout_img, in_def_img, inout_img);
   }

   std::cout<<"Calculating a Mandelbrot fractal successful\n";

   return 0;
}

