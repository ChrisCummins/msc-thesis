/*
 * Skepu Implementation of SPH (Smoothed Particle Hydrodynamics) for fluid dynamics
 * problem (shocktube simulation)
 */

// following define to enable/disable CUDA implmentation to be used
#define SKEPU_CUDA

// following define to specify number of GPUs to be used. Specifying 0 means all available GPUs. Default is 1 GPU.
/* #define SKEPU_NUMGPU 0 */

#include "sph.h"

/*****************************************
 * 	Data parallel kernels
 *****************************************/
#define s  0.006
#define cx 0.0
#define cy 0.0
#define cz 0.035
#define s2 0.001
#define s3 0.0001

GENERATE_FUNC(init, Particle, int, index, seed,

              int z = index%ZLEN;
              int y = ((index-z)/ZLEN)%YLEN;
              int x = (((index-z)/ZLEN)-y)/XLEN;

              double rand1 = (((int)(10*s*x*s2+4*cz*s3+5*y*z + 10*s*y))%RAND_MAX) - 0.5;
              double rand2 = (((int)(9*s*y*s2+3*z*s3+3*y*z*s + 5*x*y))%RAND_MAX) - 0.5;

              Particle p;

              p.id 	= index;
              p.x		= s * (x -XLEN/2)- cx + s2 * rand1;
              p.y		= s * (y - YLEN/2) - cy + s2 * rand2;
              p.z		= 0.8 * s * z - cz;
              p.vx		= 0.0;
              p.vy		= 0.0;
              p.vz		= 0.0;
              p.ax		= 0.0;
              p.ay		= 0.0;
              p.az		= 0.0;
              p.vhx	= 0.0;
              p.vhy	= 0.0;
              p.vhz	= 0.0;
              p.d		= 0.0;
              p.p		= 0.0;
              p.m		= 0.0;

              p.pool[0]	= 0;
              p.pool[1]	= 0;
              p.pool[2]	= 0;
              p.pool[3]	= 0;

              return p;
             )

ARRAY_FUNC(updatecell, Particle, parr, p,

           int i = p.id;
           p = parr[i];
           double dist = 0.0;

           for(int j = 0; j < NPARTICLES; j++)
{
if( i != j)
   {
      Particle pj = parr[j];

      dist = ((p.x-pj.x)*(p.x-pj.x) + (p.y-pj.y)*(p.y-pj.y) + (p.z-pj.z)*(p.z-pj.z));

      if(dist < (SMOOTHING_LENGTH*SMOOTHING_LENGTH))
      {
         if(p.pool[0] < GRID_CACHE-1)
         {
            p.pool[0]++;
            p.pool[p.pool[0]] = j;
         }
      }
   }
}
return p;
          )

ARRAY_FUNC(computedensity, Particle, parr, pi,

           int i = pi.id;
           pi = parr[i];
           pi.neighbours[0] = 0;

           pi.d = 0;
           pi.p = 0;

           double h2_r2 = 0.0;
           double dist  = 0.0;

           for (int j=0; j<NPARTICLES; j++)
{
if (i!=j)
   {
      Particle pj = parr[j];

      dist = ((pi.x-pj.x)*(pi.x-pj.x) + (pi.y-pj.y)*(pi.y-pj.y) + (pi.z-pj.z)*(pi.z-pj.z));

      if((dist < (SMOOTHING_LENGTH*SMOOTHING_LENGTH)) && (pi.neighbours[0] < GRID_CACHE-1))
      {
         h2_r2 = (SMOOTHING_LENGTH*SMOOTHING_LENGTH) - dist;

         pi.d += 2 * MASS * h2_r2 * h2_r2 * h2_r2;
         pi.neighbours[0]++;
         pi.neighbours[pi.neighbours[0]] = j;
      }
   }
}

if(pi.neighbours[0] < GRID_CACHE-1)
{
for (int k=pi.neighbours[0]+1; k<GRID_CACHE; k++)
   {
      pi.neighbours[k]=0;
   }
}

pi.d *= poly6_coef;
        pi.d  = (pi.d < 0.00001)?0.0:(1.0 / pi.d);
        pi.p  = STIFF * (pi.d - 1000.0);

        return pi;
          )

ARRAY_FUNC(updateforce, Particle, parr, pi,

           pi = parr[pi.id];
           pi.ax = 0.0;
           pi.ay = 0.0;
           pi.az = 0.0;

           double dist;
           double h_r;
           double grad_spiky;
           double lap_vis;
           double dist_x;
           double dist_y;
           double dist_z;
           double force_x;
           double force_y;
           double force_z;
           double vdiff_x;
           double vdiff_y;
           double vdiff_z;
           double prod;

           for (int j=1; j<pi.neighbours[0]; j++)
{
Particle pj = parr[pi.neighbours[j]];

   dist_x = pi.x - pj.x;
   dist_y = pi.y - pj.y;
   dist_z = pi.z - pj.z;

   dist = sqrt(dist_x * dist_x + dist_y * dist_y + dist_z * dist_z);

   if (dist<SMOOTHING_LENGTH*SMOOTHING_LENGTH)
      dist=SMOOTHING_LENGTH*SMOOTHING_LENGTH;

   h_r 		= 	SMOOTHING_LENGTH - dist;
   grad_spiky	=	grad_spiky_coef * pi.d * pj.d * 2 * MASS * h_r;
   lap_vis		=	lap_vis_coef * pi.d * pj.d * 2 * MASS * h_r;
   prod 		= 	(-0.5 * (pi.p + pj.p) * grad_spiky * h_r / dist);

   force_x		=	prod*dist_x;
   force_y		=	prod*dist_y;
   force_z		=	prod*dist_z;

   vdiff_x		=	(VISCOSITY*lap_vis) * (pj.vx - pi.vx);
   vdiff_y		=	(VISCOSITY*lap_vis) * (pj.vy - pi.vy);
   vdiff_z		=	(VISCOSITY*lap_vis) * (pj.vz - pi.vz);

   force_x 	+=	vdiff_x;
   force_y 	+=	vdiff_y;
   force_z		+=	vdiff_z;

   pi.ax		+=	force_x;
   pi.ay		+=	force_y;
   pi.az		+=	force_z;
}

return pi;
          )

UNARY_FUNC(updateposition, Particle, pi,

           double e		=	1.0;
           double sphere_radius= 	0.004;
           double stiff	=	30000.0;
           double damp		=	128.0;


           double col_x	=	0.0;
           double col_y	=	0.0;
           double col_z	=	0.0;
           double pre_px	=	0.0;
           double pre_py	=	0.0;
           double pre_pz	=	0.0;
           double vhx		=	0.0;
           double vhy		=	0.0;
           double vhz		=	0.0;

           pre_px		=	pi.x + TIME_STEP * pi.vhx;
           pre_py		=	pi.y + TIME_STEP * pi.vhy;
           pre_pz		=	pi.z + TIME_STEP * pi.vhz;

           vhx			=	pi.vhx + TIME_STEP * pi.ax;
           vhy			=	pi.vhy + TIME_STEP * pi.ay;
           vhz			= 	pi.vhz + TIME_STEP * pi.az;

           pi.x		=	pi.x + TIME_STEP * vhx;
           pi.y		=	pi.y + TIME_STEP * vhy;
           pi.z		=	pi.z + TIME_STEP * vhz;

           pi.vx		=	0.5 * (pi.vhx + vhx);
           pi.vy		=	0.5 * (pi.vhy + vhy);
           pi.vz		=	0.5 * (pi.vhz + vhz);

           pi.vhx		=	vhx;
           pi.vhy		=	vhy;
           pi.vhz		=	vhz;

           return pi;
          )

UNARY_FUNC(assign, Particle, pi,
           return pi;
          )

/*********************************************
 * 	Main flow (Host) with kernel calls
 *********************************************/

int main()
{

   int i, timesteps = 100;

   double time;

   struct timespec stime, etime;

   clock_gettime(CLOCK_REALTIME, &stime);

   int NTrials = 3;

   std::cout << ":::SPH Fluid Dynamics:::\n";
   std::cout << "Problem size (Number of Particles): " << XYLEN * XYLEN << "\n";

   for(int i=0; i<NTrials; i++)
   {

      skepu::Generate<init> 			sph_init(new init);
      skepu::Map<assign>			    sph_assign(new assign);
      skepu::MapArray<updatecell> 	sph_update_cell(new updatecell);
      skepu::MapArray<computedensity> sph_compute_density(new computedensity);
      skepu::MapArray<updateforce>	sph_update_force(new updateforce);
      skepu::Map<updateposition>		sph_update_position(new updateposition);

      skepu::Vector<Particle> 		fluid1(NPARTICLES);
      skepu::Vector<Particle> 		fluid2(NPARTICLES);

      sph_init(NPARTICLES, fluid1);
      sph_assign(fluid1, fluid2);

      sph_update_cell(fluid2, fluid1, fluid1);

      fluid1[0];
      fluid2[0];

      for(i=0; i<timesteps; i++)
      {
         sph_compute_density(fluid1, fluid2, fluid2);

         /* To ensure data is updated back in the host CPU memory... */
         fluid2.updateHost();

         sph_update_force(fluid2, fluid1, fluid1);

         /* To ensure data is updated back in the host CPU memory... */
         fluid1.updateHost();

         sph_update_position(fluid1, fluid1);

         /* To ensure data is updated back in the host CPU memory... */
         fluid1.updateHost();
      }

      save_step(fluid1);

      clock_gettime(CLOCK_REALTIME, &etime);

      if(i>0)
      {
         time = time + (((etime.tv_sec  - stime.tv_sec) + 1e-9*(etime.tv_nsec  - stime.tv_nsec)));
      }
   }

   std::cout << "Time taken:: " << (time/(NTrials-1)) <<" secs.\n";


}
