/*
 * map_reduce.cc - Simple Map and Reduce examples using SkelCL.
 */
#include <SkelCL/SkelCL.h>
#include <SkelCL/Vector.h>
#include <SkelCL/Map.h>
#include <SkelCL/Reduce.h>

int main(int argc, char* argv[]) {
    // Initialise SkelCL to use any device.
    skelcl::init(skelcl::nDevices(1).deviceType(skelcl::device_type::ANY));

    // Define the skeleton objects.
    skelcl::Reduce<int(int)> sum("int func(int x, int y) { return x + y; }",
                                 "0");
    skelcl::Map<int(int)> mapAdd1("int func(int x) { return x + 1; }");
    skelcl::Map<int(int)> mapDbl("int func(int x) { return 2 * x; }");
    skelcl::Map<int(int)> mapSqr("int func(int x) { return x * x; }");

    // Define two vectors A and B of length "n".
    const int n = 1000;
    skelcl::Vector<int> A(n);

    for (int i = 1; i <= n; i++) A[i - 1] = i;  // Map
    int sA = *sum(A).begin();  // Reduce
    std::cout << "The sum of the digits between ["
              << A.front() << "," << A.back() << "] is "
              << sA << "\n";

    skelcl::Vector<int> B = mapAdd1(A);  // Map
    int sB = *sum(B).begin();  // Reduce
    std::cout << "The sum of the digits between ["
              << B.front() << "," << B.back() << "] is "
              << sB << "\n";

    skelcl::Vector<int> C = mapDbl(B);  // Map
    int sC = *sum(C).begin();  // Reduce
    std::cout << "The sum of the digits between ["
              << C.front() << "," << C.back() << "] is "
              << sC << "\n";

    skelcl::Vector<int> D = mapSqr(C);  // Map
    int sD = *sum(D).begin();  // Reduce
    std::cout << "The sum of the squares of the digits between ["
              << C.front() << "," << C.back() << "] is "
              << sD << "\n";

    return 0;
}
