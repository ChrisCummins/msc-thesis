#include <cstdio>

//
// A global variable for testing with.
//
int g = 0;

//
// The test function.
//
int foo() {
    // BLOCK A: Perform 10 STORE operations.
    g = 1; g = 2; g = 3; g = 4; g = 5;
    g = 6; g = 7; g = 7; g = 9; g = 10;

    /*
     * Uncommenting this line will reduce the number of store
     * operations by < 10, since we are not executing BLOCK B. If the
     * number of store operations is reduced by 20, then we are losing
     * the instrumentation for BLOCK A.
     */
    // throw 0;

    // BLOCK B: Perform another 10 STORE operations.
    g = 11; g = 12; g = 13; g = 14; g = 15;
    g = 16; g = 17; g = 17; g = 19; g = 20;

    return 0;
}

int main(int argc, char **argv) {

    //
    // Run the test function.
    //
    try {
        foo();
    } catch (...) {
        // Ignore any exception.
    }

    //
    // Print the value of the global variable.
    // This requires 1 LOAD operation.
    //
    std::printf("g = %d\n", g);

    return 0;
}
