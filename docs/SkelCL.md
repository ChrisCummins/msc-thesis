# SkelCL

* Website: http://skelcl.uni-muenster.de
* Documentation: http://skelcl.uni-muenster.de/doc/
* Wiki: https://github.com/skelcl/skelcl/wiki
* Source code: https://github.com/skelcl/skelcl

SkelCL provides a high-level wrapper around OpenCL which aims to raise
the level of abstraction for heterogeneous programming.


## Example usage

A simple vector dot product implementation using the
[Zip](http://skelcl.uni-muenster.de/doc/classskelcl_1_1Zip_3_01Tout_07Tleft_00_01Tright_08_4.html)
and
[Reduce](http://skelcl.uni-muenster.de/doc/classskelcl_1_1Reduce_3_01T_07T_08_4.html)
skeletons:

```
#include <SkelCL/SkelCL.h>
#include <SkelCL/Vector.h>
#include <SkelCL/Zip.h>
#include <SkelCL/Reduce.h>

int main(int argc, char* argv[]) {
    // Initialise SkelCL to use any device.
    skelcl::init(skelcl::nDevices(1).deviceType(skelcl::device_type::ANY));

    // Define the skeleton objects.
    skelcl::Zip<int(int, int)> mult("int func(int x, int y) { return x * y; }");
    skelcl::Reduce<int(int)>   sum("int func(int x, int y) { return x + y; }", "0");

    // Define two vectors A and B of length "n".
    const int n = 1024;
    skelcl::Vector<int> A(n), B(n);

    // Populate A and B with random numbers.
    skelcl::Vector<int>::iterator a = A.begin(), b = B.begin();
    while (a != A.end()) {
        *a = rand() % n; ++a;
        *b = rand() % n; ++b;
    }

    // Calculate the dot product.
    int AdotB = *sum(mult(A, B)).begin();

    return 0;
}
```


## Features

TODO


## Authors

Research Group Parallel and Distributed Systems, Department of
Mathematics and Computer Science, University of Münster, Germany.

## License

Multi-license using
[GPL v3](https://raw.githubusercontent.com/skelcl/skelcl/master/LICENSE-gpl.txt)
and a (seemingly custom tailored)
[academic license](https://github.com/skelcl/skelcl/blob/master/LICENSE-academic.txt).

## Directory Overview

* `cmake-modules/` Contains the `FindOpenCL` cmake module.
* `examples/` SkelCL implementations of Dot product, Gaussian blur,
  Mandelbrot set, Matrix Multiplication and SAXPY.
* `include/CL/` Khronos' OpenCL 1.1 headers.
* `include/SkelCL/` SkelCL headers.
* `include/SkelCL/detail/` Implementation details, e.g. template
  definitions.
* `libraries/` pvsutil, and 3rd party libraries: gtest, and llvm.
* `libraries/pvsutil` Utility classes and functions, e.g. Logger, Timer.
* `libraries/stooling` TODO
* `msvc/` IDE-specific config (Microsoft Visual Studio).
* `src/` SkelCL implementation.
* `test/` Unit tests for source files.
* `xcode/` IDE-specific config (Xcode).

## Installation

Varies between different systems, but generally it uses an out-of-tree
cmake build system, with a system-specific bootstrap script to
download the necessary dependencies:

```
$ ./installDependencies<SYSTEM>.sh
$ mkdir build && cd build
$ cmake ..
$ make
```


## How It Works

TODO


## Publications

* S. Breuer, M. Steuwer, and S. Gorlatch: Extending the SkelCL
  Skeleton Library for Stencil Computations on Multi-GPU Systems. In
  Proceedings of the 1st International Workshop on High-Performance
  Stencil Computations (HiStencils), 2014, Vienna, Austria.
* C. Kessler, S. Gorlatch, J. Emmyren, U. Dastgeer, M. Steuwer, and
  P. Kegel: Skeleton Programming for Portable Many-Core
  Computing. Book chapter in Programming Multi-core and Many-core
  Computing Systems edited by S. Pllana and F. Xhafa, New York, USA,
  Wiley Interscience (to be published).
* M. Steuwer, M. Friese, S. Albers, and S. Gorlatch: Introducing and
  Implementing the Allpairs Skeleton for Programming Multi-GPU
  Systems. In International Journal of Parallel Programming, 2013,
  Springer.
* M. Steuwer and S. Gorlatch: SkelCL: Enhancing OpenCL for High-Level
  Programming of Multi-GPU Systems. In Parallel Computing
  Technologies - 12th International Conference (PaCT) Proceedings,
  2013, St. Petersburg, Russia.
* M. Steuwer and S. Gorlatch: High-Level Programming for Medical
  Imaging on Multi-GPU Systems using the SkelCL Library. In Procedia
  Computer Science, Volumne 18, Pages 749-758, 2013, Elsevier.
* P. Kegel, M. Steuwer, and, S. Gorlatch: Uniform High-Level
  Programming of Many-Core and Multi-GPU Systems. In Transition of HPC
  Towards Exascale Computing, 2013, IOS Press.
* M. Steuwer, P. Kegel, and S. Gorlatch: A High-Level Programming
  Approach for Distributed Systems with Accelerators. In New Trends in
  Software Methodologies, Tools and Techniques - Proceedings of the
  Eleventh SoMeT'12, Genoa, Italy, 2012.
* M. Steuwer, S. Gorlatch, M. Buß, and, S. Breuer: Using the SkelCL
  Library for High-Level GPU Programming of 2D Applications. In
  Euro-Par 2012: Parallel Processing Workshops, Rhodes Island, Greece,
  2012.
* M. Steuwer, P. Kegel, and S. Gorlatch: Towards High-Level
  Programming of Multi-GPU Systems Using the SkelCL Library. In 2012
  IEEE International Symposium on Parallel and Distributed Processing
  Workshops (IPDPSW), Shanghai, China, 2012.
* M. Steuwer, P. Kegel, and S. Gorlatch: SkelCL - A Portable Skeleton
  Library for High-Level GPU Programming. In 2011 IEEE International
  Symposium on Parallel and Distributed Processing Workshop and Phd
  Forum (IPDPSW), Anchorage, USA, 2011.
