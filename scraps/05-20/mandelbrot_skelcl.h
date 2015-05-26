#include <sys/time.h>

#include <SkelCL/SkelCL.h>
#include <SkelCL/IndexMatrix.h>
#include <SkelCL/Map.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>

SKELCL_COMMON_DEFINITION(
    typedef struct {      \
        unsigned char r;  \
        unsigned char g;  \
        unsigned char b;  \
    } Pixel;              \
)

std::ostream& operator<< (std::ostream& out, Pixel p)
{
    out << p.r << p.g << p.b;
    return out;
}

template <typename Iterator>
void writePPM (Iterator first, Iterator last,
               const size_t width, const size_t height,
               const std::string& filename)
{
    std::ofstream outputFile(filename.c_str());

    outputFile << "P6\n" << width << " " << height << "\n255\n";

    std::copy(first, last, std::ostream_iterator<Pixel>(outputFile));
}
