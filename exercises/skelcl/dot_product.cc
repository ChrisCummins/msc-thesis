#include <SkelCL/SkelCL.h>
#include <SkelCL/Zip.h>
#include <SkelCL/Reduce.h>
#include <SkelCL/Vector.h>

using namespace skelcl;

int main(int argc, char* argv[]) {
    // Initialise SkelCL to use any device.
    init(nDevices(1).deviceType(device_type::ANY));

    // Instantiate skeletons with user kernels.
    Zip<int(int, int)> mult("int func(int x, int y) { return x * y; }");
    Reduce<int(int)> sum("int func(int x, int y) { return x + y; }", "0");

    // Define two vectors A and B and fill with random numbers..
    Vector<int> A(1024), B(1024);
    Vector<int>::iterator a = A.begin(), b = B.begin();
    while (a != A.end()) { *a = rand() % 100; ++a; *b = rand() % 100; ++b; }

    // Call skeletons.
    Vector<int> result = sum(mult(A, B));

    // Read result.
    printf("%d\n", result.front());

    return 0;
}
