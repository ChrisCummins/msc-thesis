#include "./common.h"

// write image to a PPM file with the given filename
void WritePPM(Pixel *pixels, const char *filename, int width, int height) {
    FILE* outputFile = fopen(filename, "w");

    fprintf(outputFile, "P6\n%d %d\n255\n", width, height);
    fwrite(pixels, sizeof(Pixel), width * height, outputFile);

    fclose(outputFile);
}
