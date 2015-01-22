# SkelCL

* Website: http://skelcl.uni-muenster.de
* Documentation: http://skelcl.uni-muenster.de/doc/
* Wiki: https://github.com/skelcl/skelcl/wiki
* Source code: https://github.com/skelcl/skelcl

SkelCL provides a high-level wrapper around OpenCL which aims to raise
the level of abstraction for heterogeneous programming. It provides a
set of algorithmic skeletons for data parallel operations: Map,
Reduce, Scan, Zip, MapOverlap, and AllPairs. Each Skeleton is
parameterised with muscle functions by the user, and is compiled into
an OpenCL kernel for execution on device hardware. Communication
between the host and device memory is performed lazily and is hidden
from the user.


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
    const int n = 1024; skelcl::Vector<int> A(n), B(n);

    // Populate A and B with random numbers.
    skelcl::Vector<int>::iterator a = A.begin(), b = B.begin();
    while (a != A.end()) {
        *a = rand() % n; ++a;
        *b = rand() % n; ++b;
    }

    // Dot product: x = A . B
    int x = sum(mult(A, B)).first();

    return 0;
}
```


## Features

* Reduces OpenCL boilerplate, resulting in lower SLOC programs.
* 6 data-parallel Algorithmic Skeletons for multi-GPU execution.
* Object-orientated design with classes for Map, Reduce, Scan, Zip,
  MapOverlap, and AllPairs Skeletons.
* Supports one and two dimensional datasets using Vector and Matrix
  container classes.
* Lazy (i.e. copy on read) communication between host and device
  memory is performed automatically.
* Skeleton classes are instantiated with kernel functions represented
  as strings.


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
  Mandelbrot set, Matrix Multiplication, and SAXPY.
* `include/CL/` Khronos' OpenCL 1.1 headers.
* `include/SkelCL/` SkelCL header files. Contains files for each of
  the skeleton template classes: `AllPairs`, `Map`, `MapOverlap`,
  `Reduce`, `Scan`, and `Zip`; and data structures: `Vector`, and
  `Matrix`.
* `include/SkelCL/detail/` "Private" implementation headers,
  containing utility template classes and macros, and skeleton
  definitions.
* `libraries/` pvsutil, and 3rd party libraries: gtest, and llvm.
* `libraries/pvsutil` Utility classes and functions, e.g. Logger, Timer.
* `libraries/stooling` TODO
* `msvc/` IDE-specific config (Microsoft Visual Studio).
* `src/` SkelCL implementation code. Contains implementations for
  classess representing devices, programs, events, skeletons, etc.
* `test/` Unit tests.
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

To run the test suite: `make test`.

## How It Works

Each of the 6 Skeletons is represented it's own Header file
`include/SkelCL/<name>.h`, which defines a template class of the same
name.
[For example](https://github.com/ChrisCummins/skelcl/blob/ae14347ba48bad8a93327ef6db357e0182003d85/include/SkelCL/Reduce.h#L95):

```
template<typename T>
class Reduce<T(T)> : public detail::Skelton { /* methods declarations */ }
```

Each Skeleton class extends the class
[`detail::Skeleton`](https://github.com/ChrisCummins/skelcl/blob/ae14347ba48bad8a93327ef6db357e0182003d85/include/SkelCL/detail/Skeleton.h#L52),
which defines the common interface for all Skeletons. The main header
files declare the class methods, while the definitions themselves are
located in the header files `include/SkelCL/detail/<name>Def.h`, which
are included at foot of the public headers.
[For example](https://github.com/ChrisCummins/skelcl/blob/ae14347ba48bad8a93327ef6db357e0182003d85/include/SkelCL/Reduce.h#L221):

```
// including the definition of the templates
#include "detail/ReduceDef.h
```

These definition files contain definitions of class methods
responsible for processing input, compiling OpenCL kernels, and
preparing output.
[For example](https://github.com/ChrisCummins/skelcl/blob/ae14347ba48bad8a93327ef6db357e0182003d85/include/SkelCL/detail/ReduceDef.h#L247):

```
template <typename T>
skelcl::detail::Program Reduce<T(T)>::createPrepareAndBuildProgram()
{
  ASSERT_MESSAGE(!_userSource.empty(),
                 "Tried to create program with empty user source.");
  // first: device specific functions
  std::string s(detail::CommonDefinitions::getSource());
  // second: user defined source
  s.append(_userSource);
  // last: append skeleton implementation source
  s.append(
#include "ReduceKernel.cl"
      );

  auto program =
      detail::Program(s, skelcl::detail::util::hash("//Reduce\n" + s));
  if (!program.loadBinary()) {
    // append parameters from user function to kernels
    program.transferParameters(_funcName, 2, "SCL_REDUCE_1");
    program.transferParameters(_funcName, 2, "SCL_REDUCE_2");
    program.transferArguments(_funcName, 2, "SCL_FUNC");
    // rename user function
    program.renameFunction(_funcName, "SCL_FUNC");
    // rename typedefs
    program.adjustTypes<T>();
  }
  program.build();
  return program;
}
```

Each Skeleton has one or more associated OpenCL kernels, located in
`include/SkelCL/detail<name>Kernel.cl`. These files contain
implementations for the various kernels.
[For example](https://github.com/ChrisCummins/skelcl/blob/ae14347ba48bad8a93327ef6db357e0182003d85/include/SkelCL/detail/ReduceKernel.cl#L49):

```
// ------------------------------------ Kernel 1 -----------------------------------------------

__kernel void SCL_REDUCE_1 (
    const __global SCL_TYPE_0* SCL_IN,
          __global SCL_TYPE_0* SCL_OUT,
    const unsigned int  DATA_SIZE,
    const unsigned int  GLOBAL_SIZE)
{
    const int my_pos = get_global_id(0);
    if (my_pos > DATA_SIZE) return;

    const unsigned int modul = GLOBAL_SIZE;

    SCL_TYPE_0 res = SCL_IN[my_pos];
    int        i   = my_pos + modul;

    while ( i < DATA_SIZE )
    {
      res = SCL_FUNC( res, SCL_IN[i] );
      i = i + modul;
    }

    SCL_OUT[my_pos] = res;
}
```

A definition for the `()` operator is also provided in each Defs
header, which performs the "useful" work of processing the supplied
args, preparing the input and returning the output.
[For example](https://github.com/ChrisCummins/skelcl/blob/ae14347ba48bad8a93327ef6db357e0182003d85/include/SkelCL/detail/ReduceDef.h#L96):

```
template <typename T>
template <typename... Args>
Vector<T>& Reduce<T(T)>::operator()(Out<Vector<T>> output,
                                    const Vector<T>& input, Args&&... args)
{
  const size_t global_size = 8192;

  prepareInput(input);
  ASSERT(input.distribution().devices().size() == 1);

  // TODO: relax to multiple devices later
  auto &device = *(input.distribution().devices().front());

  Vector<T> tmpOutput;
  prepareOutput(tmpOutput, input, global_size);
  prepareOutput(output.container(), tmpOutput, 1);

  execute_first_step(device, input.deviceBuffer(device),
                     tmpOutput.deviceBuffer(device), input.size(), global_size,
                     args...);

  size_t new_data_size = std::min(global_size, input.size());

  execute_second_step(device, tmpOutput.deviceBuffer(device),
                      output.container().deviceBuffer(device), new_data_size,
                      args...);

  // ... finally update modification status.
  updateModifiedStatus(output, std::forward<Args>(args)...);

  return output.container();
}
```

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
