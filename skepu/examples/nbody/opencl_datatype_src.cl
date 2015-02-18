/* 
   When using a user-defined element type, you need to define 
   it here as well as it needs to be compiled with OpenCL
   The name of this file could be different but in that case
   you would have to specify that name in skepu/globals.h by
   specifying it OPENCL_SOURCE_FILE_NAME macros. 
*/
typedef struct _Particle
{
   double id;
   double x, y, z;
   double vx, vy, vz;
   double ax, ay, az;
   double m;
}Particle;
