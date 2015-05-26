// Source: http://rosettacode.org/wiki/Mandelbrot_set#C
#include "./common.h"

void mandelbrotSeq(Pixel* img, float startX, float startY, float dx, float dy,
                   int iterations, int width, int height) {
  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i++) {
      float x = startX + i * dx;
      float y = startY + j * dy;

      int n = 0;
      float rNext = 0.0f;
      float r = 0.0f, s = 0.0f;

      while (((r * r) + (s * s) <= 4.0f) && (n < iterations)) {
        rNext = ((r * r) - (s * s)) + x;
        s = (2 * r * s) + y;
        r = rNext;
        n++;
      }

      Pixel *p = &img[j * width + i];

      if (n == iterations) {
        p->r = 0;
        p->g = 0;
        p->b = 0;
      } else {
        p->r = COLOR_R(n);
        p->g = COLOR_G(n);
        p->b = COLOR_B(n);
      }
    }
  }
}

int main() {
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

    gettimeofday(&start, NULL);
    mandelbrotSeq(img, startX, startY, dx, dy, iterations, width, height);
    gettimeofday(&end, NULL);
    printf("Time elapsed: %f ms\n",
           (float) (1000.0 * (end.tv_sec - start.tv_sec)
                    + 0.001 * (end.tv_usec - start.tv_usec)));

    WritePPM(img, "mandelbrot_seq.ppm", width, height);
    delete[] img;

    return 0;
}
