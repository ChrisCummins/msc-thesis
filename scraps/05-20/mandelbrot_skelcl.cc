// Author: Michel Steuwer <michel.steuwer@uni-muenster.de>

#include "./mandelbrot_skelcl.h"

using namespace skelcl;

const char* kernelSource = R"(
#define COLOR_R(n) ((n & 63) << 2)
#define COLOR_G(n) ((n << 3) & 255)
#define COLOR_B(n) ((n >> 8) & 255)

Pixel func(IndexPoint position,
           float startX, float startY,
           float dx, float dy,
           const uint iterations) {
  float x = startX + position.x * dx;
  float y = startY + position.y * dy;

  int n = 0;
  float rNext = 0.0f;
  float r = 0.0f, s = 0.0f;
  while (((r * r) + (s * s) <= 4.0f) &&  (n < iterations)) {
    rNext = ((r * r) - (s * s)) + x;
    s = (2 * r * s) + y;
    r = rNext;
    n++;
  }

  Pixel pixel;
  if (n == iterations) {
    pixel.r = 0;
    pixel.g = 0;
    pixel.b = 0;
  } else {
    pixel.r = COLOR_R(n);
    pixel.g = COLOR_G(n);
    pixel.b = COLOR_B(n);
  }

  return pixel;
})";

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

    skelcl::init(skelcl::nDevices(1).deviceType(skelcl::device_type::ANY));

    IndexMatrix positions({(size_t)height, (size_t)width});
    Map<Pixel(IndexPoint)> m(kernelSource);

    gettimeofday(&start, NULL);
    Matrix<Pixel> output = m(positions, startX, startY, dx, dy, iterations);
    output.copyDataToHost();
    gettimeofday(&end, NULL);
    printf("Time elapsed: %f ms\n",
           (float) (1000.0 * (end.tv_sec - start.tv_sec)
                    + 0.001 * (end.tv_usec - start.tv_usec)));

    writePPM(output.begin(), output.end(), width, height, "mandelbrot_skelcl.ppm");

    return 0;
}
