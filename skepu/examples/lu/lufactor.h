/* LU Factorization application ported to SkePU using Generate and MapArray skeletons */

#include "skepu/vector.h"
#include "skepu/maparray.h"
#include "skepu/generate.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "math.h"
#include <time.h>

using namespace std;



#define N	PROBLEM_SIZE
#define D(i,j)	(i*N + j)

inline float getRand()
{
   float n = ceil(10*(rand()/(float)RAND_MAX-0.5));

   n = (n == 0.0)?1.00:n;

   return n;
}

void init_matrix(skepu::Vector<float> &A)
{
   int i, j;
   for(i=0; i<N; i++)
   {
      for(j=0; j<N; j++)
      {
         A[D(i,j)] = getRand();
      }
   }
}

void serial_LUFactor(skepu::Vector<float> &A)
{
   int i, j, k;

   for (k = 0; k < N; k++)
   {
      for (i = k + 1; i<N; i++)
      {
         A[D(i,k)] = A[D(i,k)]/A[D(k,k)];
      }

      for (j = k+1; j<N; j++)
      {
         for (i = k + 1; i<N; i++)
         {
            A[D(i,j)] = A[D(i,j)] - A[D(i,k)]*A[D(k,j)];
         }
      }
   }
}

void compare_results(skepu::Vector<float> &U, skepu::Vector<float> &LU)
{
   int equal = 1;
   int i,j;
   float diff = 0;

   for(i=0; i<N; i++)
   {
      for(j=0; j<N; j++)
      {
         diff = (LU[D(i,j)] - U[D(i,j)]);
         if(diff > 0.001)
         {
            equal = 0;
            break;
         }
      }
   }

   if(equal == 0)
      std::cout << "\nNot Equal\n";
   else
      std::cout << "\nResults match\n";
}

void print_matrix(skepu::Vector<float> &matrix, std::string name)
{
//    int i,j;

   matrix[0];

//    skepu::Vector<float> serial_A(N*N+1, (float)0.0);
//    serial_LUFactor(serial_A);
//
//    compare_results(serial_A, matrix);

   /*
   skepu::cout << "\nMatrix " << name << ":\n";
   for(i=0; i<N; i++)
   {
   for(j=0; j<N; j++)
   {
       skepu::cout << "\t" << matrix[D(i,j)];
   }
   skepu::cout << "\n";
      }
      */
}
