/*
 * Skepu Implementation of SPH (Smoothed Particle Hydrodynamics) for fluid dynamics
 * problem (shocktube simulation)
 */

#include "skepu/vector.h"
#include "skepu/map.h"
#include "skepu/maparray.h"
#include "skepu/generate.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "math.h"
#include <time.h>

#define XLEN 			XYLEN
#define YLEN 			XYLEN
#define ZLEN 			1
#define NPARTICLES 		(XLEN*YLEN*ZLEN)
#define SMOOTHING_LENGTH 	(1.00/NPARTICLES)
#define SEARCH_RADIUS 		(1*SMOOTHING_LENGTH)
#define MAX_FLOAT       	3.40282347e+36
#define GRID_CACHE 		10
#define MASS 			0.00020543
#define PI 			3.1415926
#define STIFF 			1.5
#define VISCOSITY 		0.2
#define TIME_STEP 		0.003
#define EPSILON 		0.00001

#define GLASS_R  		0.05
#define GLASS_BOTTOM  	       -0.08
#define GLASS_TOP  		0.06
#define GLASS_THICKNESS  	0.01

#define poly6_coef		(315.0/(64.0*PI*pow(SMOOTHING_LENGTH,9)))
#define grad_poly6_coef		(945.0/(32.0*PI*pow(SMOOTHING_LENGTH,9)))
#define lap_poly6_coef		(945.0/(32.0*PI*pow(SMOOTHING_LENGTH,9)))
#define grad_spiky_coef		(-45.0/(PI*pow(SMOOTHING_LENGTH,6)))
#define lap_vis_coef		(45.0/(PI*pow(SMOOTHING_LENGTH,6)))

struct Particle
{
   int id;			// id of the particle
   double x,y,z;		// coordinate of the particle
   double vx,vy,vz;		// velocity of the particle
   double vhx,vhy,vhz;		// velocity half
   double ax,ay,az;		// acceleration of particle
   double m;			// mass of the particle
   double p;			// pressure of the particle
   double d;			// density of particle
   int pool[GRID_CACHE];	// pool
   int neighbours[GRID_CACHE]; // neighbours list
};


// Utility Functions
void save_step(skepu::Vector<Particle> &fluid)
{
   int j = 0;

   //for(j=0; j<fluid.size(); j++)
   {
      Particle p=fluid[j];

      //printf("%f\t%f\t%f\t", p.x, p.y, p.z);

      std::cout<<std::fixed<<std::setprecision(6)<<p.x<<"\t"
               <<std::fixed<<std::setprecision(6)<<p.y<<"\t"
               <<std::fixed<<std::setprecision(6)<<p.z<<"\t";

      /* printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t", p.x,p.y,p.z,
      							  p.vx,p.vy,p.vz,
      							  p.ax,p.ay,p.az,
      							  p.d, p.p, p.m); */
   }

   std::cout << "\n";
}