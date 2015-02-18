// following define to enable/disable OpenMP implmentation to be used
/* #define SKEPU_OPENMP */

// following define to enable/disable OpenCL implmentation to be used
/* #define SKEPU_OPENCL */
// With OpenCL, following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include "lufactor.h"


ARRAY_FUNC(factorize, float, A, ind,
           int index = (int)(ind-1);
           if(index > N*N)
           return A[index];
           int col = index%N;
           int row = ((index-col)/N)%N;
           int iteration = (int)A[N*N];
           int LorU = (int)10*(A[N*N] - (float)iteration);

           if(index == N*N)
{
   if(LorU == 0)
         return (A[index] + 0.1f);

      return (iteration + 1.0f);
   }

if(LorU == 0)
{
if((col == iteration) && (row > col))
   {
      return A[index]/A[D(col,col)];
   }
}
else if(LorU == 1)
{
if( (col > iteration) && (row > iteration) )
   {
      return (A[index] - A[D(row,iteration)]*A[D(iteration,col)]);
   }
}

return A[index];
          )

GENERATE_FUNC(indexer, int, int, index, seed,
              return index+1;
             )

void skepu_factorize(skepu::Vector<float>& A, skepu::Vector<float>& LU)
{
   skepu::Generate<indexer> set_indices(new indexer);
   skepu::MapArray<factorize> lu_factorize(new factorize);

   skepu::Vector<float> indices(N*N+1, (float)0.0);

   set_indices(N*N+1, indices);

   for(int i=0; i<N; i++)
   {
      lu_factorize(A, indices, LU);
      lu_factorize(LU, indices, A);
   }
}

int main(int argc, char **argv)
{
   struct timespec stime, etime;

   double time = 0.0f;
   int NTrials = 3;

   std::cout << ":::LU Factorization:::\n";
   std::cout << "Problem size (Matrix Dimension): " << N << "\n";

   for(int i=0; i<NTrials; i++)
   {
      skepu::Vector<float> A(N*N+1, (float)0.0);
      skepu::Vector<float> B;
      skepu::Vector<float> LU(N*N+1, (float)0.0);

      init_matrix(A);

      B = A;

      clock_gettime(CLOCK_REALTIME, &stime);
      skepu_factorize(A, LU);
      A.flush();

//	print_matrix(LU, "LU");
      clock_gettime(CLOCK_REALTIME, &etime);

      if(i>0)
      {
         time = time + (((etime.tv_sec  - stime.tv_sec) + 1e-9*(etime.tv_nsec  - stime.tv_nsec)));
      }
   }

   std::cout << "Time taken:: " << (time/(NTrials-1)) <<" secs, N = "<<N<<".\n";

   return 0;
}
