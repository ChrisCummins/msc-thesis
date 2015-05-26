// Author: Michel Steuwer <michel.steuwer@uni-muenster.de>

#include "./common.h"

// ######################################################
// Start OpenCL section
cl_platform_id    platform;
cl_device_id      device;
cl_context        context;
cl_command_queue  commandQueue;
cl_kernel         kernel;

// check err for an OpenCL error code
void checkError(cl_int err) {
  if (err != CL_SUCCESS)
    printf("Error with errorcode: %d\n", err);
}

void initOpenCL() {
  cl_int err;

  // Speichere 1 Plattform in platform
  err = clGetPlatformIDs(1, &platform, NULL);
  checkError(err);
  printf("platform selected\n");

  // Speichere 1 Device beliebigen Typs in device
  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  checkError(err);
  printf("device selected\n");

  // erzeuge Context fuer das Device device
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  checkError(err);
  printf("context created\n");

  // erzeuge Command Queue zur Verwaltung von device
  commandQueue = clCreateCommandQueue(context, device, 0, &err);
  checkError(err);
  printf("commandQueue created\n");
}

void printBuildLog(cl_program program, cl_device_id device) {
  cl_int err;
  char *build_log;
  size_t build_log_size;
  // Speichere den Build Log fuer program und device in build_log
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_log_size);
  checkError(err);
  build_log = (char*) malloc(build_log_size);
  err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_log_size, build_log, NULL);
  printf("Log:\n%s\n", build_log);
  free(build_log);
}

void makeKernel() {
  cl_int err;
  // Kernel Quellcode
  const char* kernelSource = R"(
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store: enable
#define COLOR_R(n) ((n & 63) << 2)
#define COLOR_G(n) ((n << 3) & 255)
#define COLOR_B(n) ((n >> 8) & 255)

typedef struct {
  unsigned char r;
  unsigned char g;
  unsigned char b;
} Pixel;

__kernel
void mandelbrotKernel(__global Pixel* pixel,
                      float startX,
                      float startY,
                      float dx,
                      float dy,
                      int iterations,
                      int width) {
  float x = startX + get_global_id(0) * dx;
  float y = startY + get_global_id(1) * dy;

  int n = 0;
  float rNext = 0.0f;
  float r = 0.0f, s = 0.0f;
  while (((r * r) + (s * s) <= 4.0f) &&  (n < iterations)) {
    rNext = ((r * r) - (s * s)) + x;
    s = (2 * r * s) + y;
    r = rNext;
    n++;
  }

  __global Pixel* p = &pixel[get_global_id(1) * width + get_global_id(0)];
  if (n == iterations) {
    p->r = 0;
    p->g = 0;
    p->b = 0;
  } else {
    p->r = COLOR_R(n);
    p->g = COLOR_G(n);
    p->b = COLOR_B(n);
  }
})";
  // Laenge des Kernel Quellcodes
  size_t sourceLength = strlen(kernelSource);
  cl_program program;
  // Ein Programm aus dem Kernel Quellcode wird erzeugt
  program = clCreateProgramWithSource(context, 1, &kernelSource, &sourceLength, &err);
  checkError(err);
  printf("program created\n");
  // Das Programm wird fuer alle Devices des Contextes gebaut
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  checkError(err);
  if (err != CL_SUCCESS)
    printBuildLog(program, device);
  else
    printf("program build successfully\n");
  kernel = clCreateKernel(program, "mandelbrotKernel", &err);
  checkError(err);
  printf("kernel created\n");
}

void mandelbrotOpenCL(Pixel* img, float startX, float startY, float dx, float dy, int iterations, int width, int height) {
  cl_int err;
  int size = width * height * sizeof(Pixel);

  // Speicher fuer Ergebnis Matrix reservieren
  cl_mem imgd = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, size, NULL, &err);
  checkError(err);
  printf("buffer imgd created and memory allocated\n");

  // Setze Argument fuer den Kernel
  err  = clSetKernelArg( kernel, 0, sizeof(cl_mem), &imgd   );
  err |= clSetKernelArg( kernel, 1, sizeof(float),  &startX );
  err |= clSetKernelArg( kernel, 2, sizeof(float),  &startY );
  err |= clSetKernelArg( kernel, 3, sizeof(float),  &dx     );
  err |= clSetKernelArg( kernel, 4, sizeof(float),  &dy     );
  err |= clSetKernelArg( kernel, 5, sizeof(int),    &iterations);
  err |= clSetKernelArg( kernel, 6, sizeof(int),    &width  );
  checkError(err);
  printf("kernel arguments set\n");

  size_t globalSize[] = {(size_t)width, (size_t)height};
  // Starte Kernel width * width mal
  err = clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL,
                               globalSize, NULL, 0, NULL, NULL);
  checkError(err);
  printf("enqueued kernel\n");

  // Daten vom Device kopieren
  // Dieser Aufruf ist blockierend (CL_TRUE)
  err = clEnqueueReadBuffer( commandQueue, imgd,  CL_TRUE, 0, size, img, 0, NULL, NULL );
  checkError(err);
  printf("enqueued read buffer imgd\n");
}

// end OpenCL section
// ######################################################

int main(void) {
  struct timeval start, end;
  int width  = 1024*4;
  int height = 768*4;
  int zoom   = 1000;
  float startX = -static_cast<float>(width)  / (zoom * 2.0);
  float endX   =  static_cast<float>(width)  / (zoom * 2.0);
  float startY = -static_cast<float>(height) / (zoom * 2.0);
  float endY   =  static_cast<float>(height) / (zoom * 2.0);
  float dx     =  (endX - startX) / width;
  float dy     =  (endY - startY) / height;
  int iterations = 2000;

  Pixel* img = new Pixel[width * height];

  initOpenCL();
  makeKernel();

  gettimeofday(&start, NULL);
  mandelbrotOpenCL(img, startX, startY, dx, dy, iterations, width, height);
  gettimeofday(&end, NULL);
  printf("Time elapsed: %f ms\n",
         (float) (1000.0 * (end.tv_sec - start.tv_sec)
                  + 0.001 * (end.tv_usec - start.tv_usec)));

  WritePPM(img, "mandelbrot_opencl.ppm", width, height);
  delete[] img;

  return 0;
}
