/*!
 *  PPMCC stands for "Pearson product-moment correlation coefficient"
 *  In statistics, the Pearson coefficient of correlation is a measure by the
 *  linear dependence between two variables X and Y. The mathematical
 *  expression of the Pearson coefficient of correlation is as follows:
 *   r = ( (n*sum(X.Y)-sum(X)*sum(Y))/((n*sum(X^2)-(sum(X))^2)*(n*sum(Y^2)-(sum(Y))^2)) )
 */

// following define to enable/disable OpenMP implmentation to be used
/* #define SKEPU_OPENMP */

// following define to enable/disable OpenCL implmentation to be used
/* #define SKEPU_OPENCL */
// With OpenCL, following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>
#include <math.h>

#include "skepu/vector.h"
#include "skepu/map.h"
#include "skepu/reduce.h"
#include "skepu/mapreduce.h"

// Unary user-function used for mapping
UNARY_FUNC(square_f, float, a,
           return a*a;
          )

// Binary user-function used for mapping
BINARY_FUNC(mult_f, float, a, b,
            return a*b;
           )

// User-function used for reduction
BINARY_FUNC(plus_f, float, a, b,
            return a+b;
           )

#define N 50

int main()
{
// vector operands...
   skepu::Vector<float> X(N);
   skepu::Vector<float> Y(N);

   X.randomize(1.0, 3.0);
   Y.randomize(2.0, 4.0);

// skeleton definitions...
   skepu::Reduce<plus_f> sumReduce(new plus_f);
   skepu::MapReduce<mult_f, plus_f> dotProduct(new mult_f, new plus_f);
   skepu::MapReduce<square_f, plus_f> sumSquare(new square_f, new plus_f);


// skeleton calls...
   float sumX = sumReduce(X);
   float sumY = sumReduce(Y);
   float sumPr = dotProduct(X,Y);
   float sumSqX = sumSquare(X);
   float sumSqY = sumSquare(Y);

// result calculation...
   float res = ((N * sumPr - sumX * sumY) / sqrt((N * sumSqX - pow(sumX, 2)) * (N * sumSqY - pow(sumY, 2))) );

   std::cout<<"X: " <<X <<"\n";
   std::cout<<"Y: " <<Y <<"\n";

   std::cout<<"res: " <<res <<"\n";

   return 0;
}
