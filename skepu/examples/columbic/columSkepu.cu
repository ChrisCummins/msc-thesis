// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

// following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include <iostream>
#include <math.h>

#include "skepu/vector.h"
#include "skepu/matrix.h"
#include "skepu/maparray.h"
#include "skepu/testing.h"

#define RAND_MAX_LOC 100

typedef struct _CoulombConst
{
   int n_atoms;
   float gridspacing;
} CoulombConst;



// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.
// Note: this implementation uses precomputed and unrolled
// loops of dy*dy + dz*dz values for increased FP arithmetic intensity
// per FP load.  The X coordinate portion of the loop is unrolled by
// four, allowing the same dy^2 + dz^2 values to be reused four times,
// increasing the ratio of FP arithmetic relative to FP loads, and
// eliminating some redundant calculations.
// This version implement's Kahan's compensated summation method to
// increase doubleing point accuracy for the large number of summed
// potential values.
//
// NVCC -cubin says this implementation uses 24 regs, 28 smem
// Profiler output says this code gets 50% warp occupancy
//
ARRAY_FUNC_MATR_CONSTANT(coulombPotential_f, double, CoulombConst, atominfo, energygridItem, coulConst, xindex, yindex,
                         int numatoms = coulConst.n_atoms;
                         double curenergyx1 = energygridItem;
                         float coory = coulConst.gridspacing * yindex;
                         float coorx1 = coulConst.gridspacing * (xindex);
                         double energyvalx1=0.0f;
                         double energycomp1=0.0f;
                         int atomid;
                         for (atomid=0; atomid<numatoms*4; atomid+=4)
{
double dy = coory - atominfo[atomid+2];
   double dysqpdzsq = (dy * dy) + atominfo[atomid+3];
   double dx1 = coorx1 - atominfo[atomid+1];
   double s;
   double y;
   double t;
   s = atominfo[atomid] * (1.0f / sqrtf(dx1*dx1 + dysqpdzsq));
   y = s - energycomp1;
   t = energyvalx1 + y;
   energycomp1 = (t - energyvalx1)  - y;
   energyvalx1 = t;
}
return curenergyx1 + energyvalx1;
                        )




//############
dim3 volsize;
double gridspacing;




/*!
 * Function to initialize atoms
 */
int initatoms(skepu::Vector<double> &atombuf, int count, dim3 volsize, double gridspacing)
{
   srand(0);

   dim3 size;
   int i;

   // compute grid dimensions in angstroms
   size.x = gridspacing * volsize.x;
   size.y = gridspacing * volsize.y;
   size.z = gridspacing * volsize.z;

   for (i=0; i<count; i+=4)
   {
      int addr = i;
      atombuf[addr    ] = (rand() / (double) RAND_MAX_LOC) * size.x;
      atombuf[addr + 1] = (rand() / (double) RAND_MAX_LOC) * size.y;
      atombuf[addr + 2] = (rand() / (double) RAND_MAX_LOC) * size.z;
      atombuf[addr + 3] = ((rand() / (double) RAND_MAX_LOC) * 2.0) - 1.0;  // charge
   }
   return 0;
}


int matrixSize=10;
int atomcount = 1000; // problem size

skepu::MapArray<coulombPotential_f> columbicPotential(new coulombPotential_f);


void runTests(skepu::Vector<double> &atombuf)
{
   skepu::Timer timer;

   CoulombConst coulConst;
   coulConst.n_atoms = atomcount;
   coulConst.gridspacing = gridspacing;

   columbicPotential.setConstant(coulConst);

   skepu::Matrix<double> grid_in(matrixSize,matrixSize);
   skepu::Matrix<double> energy_out(matrixSize,matrixSize);

   timer.start();
   columbicPotential(atombuf, grid_in, energy_out);
   timer.stop();

   // can print and compare. output is exactly same for cpu, openmp, cuda for 1 and 2 gpus.
//   	std::cout<<"energy_out: "<<energy_out<<"\n";

   double atomevalssec = ((double) volsize.x * volsize.y * volsize.z * atomcount)/ (timer.getAverageTime() * 1000000000.0);
   std::cout << "Efficiency metric, " << atomevalssec << " billion atom evals per second\n";

   /* 10 FLOPS per atom eval */
   printf("FP performance: %g GFLOPS\n", atomevalssec * 10.0);
}


int main(int argc, char** argv)
{
   skepu::Vector<double> atoms(atomcount*4);

   volsize.x = 2048;
   volsize.y = 2048;
   volsize.z = 1;

   gridspacing = 0.1;

   // allocate and initialize atom coordinates and charges
   if (initatoms(atoms, atomcount*4, volsize, gridspacing))
      return -1;

   runTests(atoms);

   return 0;
}



