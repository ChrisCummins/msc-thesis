// following define to enable/disable OpenMP implmentation to be used
/* #define SKEPU_OPENMP */

// following define to enable/disable OpenCL implmentation to be used
/* #define SKEPU_OPENCL */

// Enable following to specify how many GPU threads per block should be used with OpenCL.
// by default, SKEPU used max threads possible which could be a problem if a kernel
// consumes lot of registers and thus cannot execute with that # of threads.
/* #define SKEPU_MAX_GPU_THREADS 512 */

#include <iostream>
#include <fstream>
#include <iomanip>

#include "math.h"

#include "skepu/vector.h"
#include "skepu/maparray.h"
#include "skepu/generate.h"
#include "skepu/testing.h"

/*
   Particle data structure that is used as an element type.
   A user-defined element type in SkePU must have a default
   (zero-argument) constructor.
   Moreover, when using OpenCL backend, the type must also
   be declared in a separate file (named 'opencl_datatype_src.cl')
   as for OpenCL we need to compile type definition for any
   user-defined type.
*/
struct Particle
{
   double id;
   double x, y, z;
   double vx, vy, vz;
   double ax, ay, az;
   double m;
};

// some parameter constants. can change here....
#define NP 5 // Number of particles in 1 direction
#define G 1
#define time_steps 10
#define delta_t 0.1



/*
   Array user-function that is used for applying nbody computation,
   All elements from parr and a single element (named 'p_1') are accessible
   to produce one output element of the same type.
*/
ARRAY_FUNC(move, Particle, parr, p_1,
           int i = p_1.id;
           p_1.ax = 0.0;
           p_1.ay = 0.0;
           p_1.az = 0.0;

           double rij = 0;
           double dum = 0;

           for(int j=0; j<NP*NP*NP; ++j)
{
if(i!=j)
   {
      Particle pj = parr[j];

      rij = sqrt((p_1.x-pj.x)*(p_1.x-pj.x) + (p_1.y-pj.y)*(p_1.y-pj.y) + (p_1.z-pj.z)*(p_1.z-pj.z));

      dum = G * (pj.m) / pow(rij,3);

      p_1.ax =  p_1.ax + dum * (p_1.x-pj.x);
      p_1.ay =  p_1.ay + dum * (p_1.y-pj.y);
      p_1.az =  p_1.az + dum * (p_1.z-pj.z);
   }
}

p_1.x = parr[i].x + delta_t * parr[i].vx + ((delta_t*delta_t)/2)*(parr[i].ax);
        p_1.y = parr[i].y + delta_t * parr[i].vy + ((delta_t*delta_t)/2)*(parr[i].ay);
        p_1.z = parr[i].z + delta_t * parr[i].vz + ((delta_t*delta_t)/2)*(parr[i].az);

        p_1.vx = parr[i].vx + (delta_t/2)*(parr[i].ax + p_1.ax);
        p_1.vy = parr[i].vy + (delta_t/2)*(parr[i].ay + p_1.ay);
        p_1.vz = parr[i].vz + (delta_t/2)*(parr[i].az + p_1.az);

        return p_1;
          )



/*
   Generate user-function that is used for initializing particles array.
*/
GENERATE_FUNC(init, Particle, int, index, seed,
              int s = index;
              int d = NP/2+1;
              int i = s%NP;
              int j = ((s-i)/NP)%NP;
              int k = (((s-i)/NP)-j)/NP;

              Particle p;

              p.id = s;

              p.x = i-d+1;
              p.y = j-d+1;
              p.z = k-d+1;

              p.vx = 0.0;
              p.vy = 0.0;
              p.vz = 0.0;
              p.ax = 0.0;
              p.ay = 0.0;
              p.az = 0.0;

              p.m = 1;

              return p;
             )

/*!
 * A helper function to write particle output values to a file.
 */
void save_step(skepu::Vector<Particle> &particles, const std::string &filename)
{
   std::ofstream out(filename.c_str());

   if(!out.is_open())
   {
      std::cerr<<"Error: cannot open this file: "<<filename<<"\n";
      return;
   }
   for(int j=0; j<NP*NP*NP; j++)
   {
      Particle p=particles[j];

      out<<std::setw(15)<<p.id<<std::setw(15)<<p.x<<std::setw(15)<<p.y<<std::setw(15)<<p.z<<std::setw(15)<<p.ax<<std::setw(15)<<p.ay<<std::setw(15)<<p.az<<std::setw(15)<<p.vx<<std::setw(15)<<p.vy<<std::setw(15)<<p.vz<<"\n";
   }

   out.close();
}


/*!
 * A helper function to write particle output values to standard output stream.
 */
void save_step(skepu::Vector<Particle> &particles)
{
   for(int j=0; j<(NP*NP*NP); j++)
   {
      Particle p=particles[j];

      std::cout<<std::setw(15)<<p.id<<std::setw(15)<<p.x<<std::setw(15)<<p.y<<std::setw(15)<<p.z<<std::setw(15)<<p.ax<<std::setw(15)<<p.ay<<std::setw(15)<<p.az<<std::setw(15)<<p.vx<<std::setw(15)<<p.vy<<std::setw(15)<<p.vz<<"\n";
   }
}



int main()
{
   double t;

// Skeleton definition....
   skepu::Generate<init> nbody_init(new init);
   skepu::MapArray<move> nbody_simulate_step(new move);

// Particle vectors....
   skepu::Vector<Particle> particles((NP*NP*NP));
   skepu::Vector<Particle> latest((NP*NP*NP));

   skepu::Timer timer;
   bool time = false;

// particle vectors initialization...
   nbody_init.setConstant((int)0);
   nbody_init((NP*NP*NP), particles);
   nbody_init((NP*NP*NP), latest);

// main nbody loop...
   for(t=0; t<time_steps/2; t=t+delta_t)
   {
      if(time)
         timer.start();

      nbody_simulate_step(particles, latest, latest);
      save_step(latest, "output.txt");

      nbody_simulate_step(latest, particles, particles);
      save_step(particles, "output.txt");

      if(time)
         timer.stop();
      else
         time = true;
   }

   std::cerr<<"> Time taken (ms): "<<timer.getAverageTime()<<"\n";

   save_step(latest, "output.txt");

   return 0;
}
