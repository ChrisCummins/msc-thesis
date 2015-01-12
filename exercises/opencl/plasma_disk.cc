/* plasma_disk.cpp
*/
// #define __NO_STD_VECTOR // Use cl::vector instead of STL version
#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "opencl_error.h"

int NUMPART=256;
using namespace cl;

#ifdef GPU
int GROUP_SIZE=256;
#else
int GROUP_SIZE=1;
#endif

class PropagateNBodySystem
{
private:
   Context      *context;
   CommandQueue *queue;
   Kernel       *kernel_eom;
   Program      *program_eom;

   Buffer *gposold; // compute device: electron positions
   Buffer *gvelold; // compute device: electron velocities
   Buffer *gposnew; // compute device: store temp electron positions
   Buffer *gvelnew; // compute device: store temp electron velocities

   float *hposold;
   float *hvelold;
   float *hposnew;
   float *hvelnew;
public:
   PropagateNBodySystem(void);
   void Initialize();
   void InitializeOpenCL(bool talky);
   void PropagateDoubleStep(float DeltaTime);
   void PutElectrons(int NumBodies, float *hposnew, float *hvelnew );
   void GetElectrons(int NumBodies, float *hposnew, float *hvelnew );
   void WriteState(const char* filename);
   void RunSimulation( void );
};

PropagateNBodySystem::PropagateNBodySystem(void)
{
//
}

void PropagateNBodySystem::Initialize()
{
   hposold  =new float[4*NUMPART];
   hvelold  =new float[4*NUMPART];
   hposnew  =new float[4*NUMPART];
   hvelnew  =new float[4*NUMPART];

   srandom(0);

   for(int ip=0;ip<NUMPART;ip++)
   {
      // generate two random variables
      float rand_zerotoone=float(rand())/float(RAND_MAX);
      float rand_zeroto2pi=float(rand())/float(RAND_MAX)*M_PI*2.0;

      hposold[ip*4+0]=sqrt(2.0*NUMPART*rand_zerotoone)*cos(rand_zeroto2pi);
      hposold[ip*4+1]=sqrt(2.0*NUMPART*rand_zerotoone)*sin(rand_zeroto2pi);
      hposold[ip*4+2]=0.0f;
      hposold[ip*4+3]=1.0f;
   }
}

void PropagateNBodySystem::InitializeOpenCL( bool talky )
{
   try
   {
      // Get available platforms
      std::vector<Platform> platforms;
      Platform::get(&platforms);
      // Select the default platform and create a context using this platform and the GPU
      cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
#ifdef GPU
      context=new Context( CL_DEVICE_TYPE_GPU, cps);
#else
      context=new Context( CL_DEVICE_TYPE_CPU, cps);
#endif
      // Get a list of devices on this platform
      std::vector<Device> devices = context->getInfo<CL_CONTEXT_DEVICES>();
      // Print out information about the device
      if (talky)
      {
         #include "print_info.h"
      }
      // Create a command queue and use the first device
      queue=new CommandQueue(*context,devices[0]);

      // Read source file
      std::ifstream sourceFile_eom("integrate_eom_kernel.cl");
      std::string sourceCode_eom(std::istreambuf_iterator<char>(sourceFile_eom),(std::istreambuf_iterator<char>()));
      Program::Sources source_eom(1, std::make_pair(sourceCode_eom.c_str(), sourceCode_eom.length()+1));
      // Make program of the source code in the context
      program_eom = new Program(*context, source_eom);
      // Build program for these specific devices, compiler argument to include local directory for header files (.h)
      program_eom->build(devices,"-I.");
      // Make kernel
      kernel_eom=new Kernel(*program_eom, "integrate_eom");
      fprintf(stderr,"integrate_eom done\n");

      // Create memory buffers on GPU and populate them with the initial data
      gposold  = new Buffer(*context, CL_MEM_READ_WRITE, 4*NUMPART * sizeof(float));
      gvelold  = new Buffer(*context, CL_MEM_READ_WRITE, 4*NUMPART * sizeof(float));
      gposnew  = new Buffer(*context, CL_MEM_READ_WRITE, 4*NUMPART * sizeof(float));
      gvelnew  = new Buffer(*context, CL_MEM_READ_WRITE, 4*NUMPART * sizeof(float));
      queue->enqueueWriteBuffer(*gposold, CL_TRUE, 0, 4*NUMPART * sizeof(float), hposold);
      queue->enqueueWriteBuffer(*gvelold, CL_TRUE, 0, 4*NUMPART * sizeof(float), hvelold);
   }
   catch(Error error)
   {
      std::cerr << error.what() << "(" << oclErrorString(error.err()) << ")" << std::endl;
      exit(1);
   }
}

void PropagateNBodySystem::PropagateDoubleStep( float DeltaTime )
{
   try
   {
      // Set arguments to kernel
      kernel_eom->setArg( 0, *gposold);
      kernel_eom->setArg( 1, *gvelold);
      kernel_eom->setArg( 2, *gposnew);
      kernel_eom->setArg( 3, *gvelnew);
      kernel_eom->setArg( 4, Local(GROUP_SIZE*4*sizeof(float)));
      kernel_eom->setArg( 5, NUMPART);
      kernel_eom->setArg( 6, DeltaTime);

      // Run the kernel on specific ND range
      NDRange global(NUMPART);
      NDRange local(GROUP_SIZE);

      Event eventA;
      queue->enqueueNDRangeKernel(*kernel_eom, NullRange, global, local,0,&eventA);
      eventA.wait();

      // set pointers to particles to kernel in reverse order to copy new->old

      kernel_eom->setArg(0, *gposnew);
      kernel_eom->setArg(1, *gvelnew);
      kernel_eom->setArg(2, *gposold);
      kernel_eom->setArg(3, *gvelold);

      Event eventB;
      queue->enqueueNDRangeKernel(*kernel_eom, NullRange, global, local,0,&eventB);
      eventB.wait();
   }
   catch(Error error)
   {
      std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
      exit(1);
   }
}

void PropagateNBodySystem::WriteState(const char* filename)
{
   // transfer memory back to CPU
   queue->enqueueReadBuffer(*gposold, CL_TRUE, 0, 4*NUMPART*sizeof(float), hposold);
   queue->enqueueReadBuffer(*gvelold, CL_TRUE, 0, 4*NUMPART*sizeof(float), hvelold);
   FILE *fd=fopen(filename,"w");
   for(int i=0;i<NUMPART;i++)
   {
      fprintf(fd,"%e %e %e %e"   ,hposold[i*4+0],hposold[i*4+1],hposold[i*4+2],hposold[i*4+3]);
      fprintf(fd," %e %e %e %e\n",hvelold[i*4+0],hvelold[i*4+1],hvelold[i*4+2],hvelold[i*4+3]);
   }
   fclose(fd);
}

void PropagateNBodySystem::GetElectrons(int NumBodies, float *pos, float *vel )
{
   // transfer memory from GPU to CPU
   queue->enqueueReadBuffer(*gposold, CL_TRUE, 0, 4*NumBodies*sizeof(float), pos);
   queue->enqueueReadBuffer(*gvelold, CL_TRUE, 0, 4*NumBodies*sizeof(float), vel);
   queue->finish();
}

void PropagateNBodySystem::PutElectrons(int NumBodies, float *pos, float *vel )
{
   // transfer memory from CPU to GPU
   queue->enqueueWriteBuffer(*gposold, CL_TRUE, 0, 4*NumBodies * sizeof(float), pos);
   queue->enqueueWriteBuffer(*gvelold, CL_TRUE, 0, 4*NumBodies * sizeof(float), vel);
   queue->finish();
}

void PropagateNBodySystem::RunSimulation( void )
{
   float deltaTime=1.0e-3f;
   int nt=1001;

   time_t c0,c1;

   Initialize();
   InitializeOpenCL(true);
   PutElectrons(NUMPART,hposold,hvelold);

   time(&c0);
   for(int it=0;it<nt;it++) // main propagation loop
   {
      if (it%100==0)
      {
         char filename[500];
#ifdef GPU
         sprintf(filename,"gpu_state%05d.dat",it);
#else
         sprintf(filename,"cpu_state%05d.dat",it);
#endif
         WriteState(filename);
      }
      PropagateDoubleStep(deltaTime);
   }
   time(&c1);
#ifdef GPU
   fprintf(stderr,"GPU propagation for %d particles over %d doublesteps took %.0f s\n",NUMPART,nt,difftime(c1,c0));
   FILE *fd=fopen("gpu_stats.dat","a");
   fprintf(fd,"%d %d %d %f\n",NUMPART,GROUP_SIZE,nt,difftime(c1,c0));
   fclose(fd);
#else
   fprintf(stderr,"CPU propagation for %d particles over %d doublesteps took %.0f s\n",NUMPART,nt,difftime(c1,c0));
   FILE *fd=fopen("cpu_stats.dat","a");
   fprintf(fd,"%d %d %d %f\n",NUMPART,GROUP_SIZE,nt,difftime(c1,c0));
   fclose(fd);
#endif
}

int main( int argc, char **argv )
{
   if (argc==3)
   {
      GROUP_SIZE=atoi(argv[1]);
      NUMPART=GROUP_SIZE*atoi(argv[2]);
   }

   PropagateNBodySystem Plasma;
   Plasma.RunSimulation();

   return 0;
}
