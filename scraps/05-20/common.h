#ifndef SCRAPS_05_20_COMMON_H_
#define SCRAPS_05_20_COMMON_H_

#include <CL/cl.h>

#include <string.h>
#include <sys/time.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>

#define COLOR_R(n) ((n & 63) << 2)
#define COLOR_G(n) ((n << 3) & 255)
#define COLOR_B(n) ((n >> 8) & 255)

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Pixel;

void WritePPM(Pixel *pixels, const char *filename, int width, int height);

#endif  // SCRAPS_05_20_COMMON_H_
