/*
 * dot_product.cc - Simple SkelCL dot product.
 *
 * Based on SkelCL example program:
 *   Author: Michel Steuwer <michel.steuwer@uni-muenster.de>
 *   License: GPL v3
 */
#include <SkelCL/SkelCL.h>
#include <SkelCL/Vector.h>
#include <SkelCL/Zip.h>
#include <SkelCL/Reduce.h>

#define UNUSED(x) (void)(x)

unsigned int _seed = 0;
unsigned int *seed = &_seed;

int main(int argc, char* argv[]) {
    // Initialise SkelCL to use any device.
    skelcl::init(skelcl::nDevices(1).deviceType(skelcl::device_type::ANY));

    // Define the skeleton objects.
    skelcl::Zip<int(int, int)> mult("int func(int x, int y) { return x * y; }");
    skelcl::Reduce<int(int)>   sum("int func(int x, int y) { return x + y; }",
                                   "0");

    // Define two vectors A and B of length "n".
    const int n = 1024;
    skelcl::Vector<int> A(n), B(n);

    // Populate A and B with random numbers.
    skelcl::Vector<int>::iterator a = A.begin(), b = B.begin();
    while (a != A.end()) {
        *a = rand_r(seed) % n; ++a;
        *b = rand_r(seed) % n; ++b;
    }

    // Calculate the dot product.
    int x = *sum(mult(A, B)).begin();

    return x ^ x;
}
