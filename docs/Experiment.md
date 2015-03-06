# E1: Characterisation of SkelCL performance

The goal is to collect data about SkelCL performance under various
conditions (independent variables) which will be used to identify an
*optimisation space*.

#### Benchmarks

###### Real world programs
  * Canny edge detection
  * FDTD
  * Game of Life
  * ~~Gaussian blur~~
  * ~~Heat simulation~~
  * ~~Mandelbrot set~~
  * ~~Sobel edge detection~~

###### Standard algorithms
  * ~~Dot product~~
  * ~~Matrix multiply~~
  * ~~SAXPY~~

#### Benchmark properties
* Skeleton: {~~Map,Reduce,Scan,Zip,AllPairs,~~Stencil}.
* Size of input data.
* Boundary size: [1,10].
* ~~OpenCL kernel properties: size of user functions, number of
  branches, number of loads/stores.~~
* Program: (enumerated list of all benchmarks).
* Device properties: number of cores, amount of memory, memory
  bandwidth.

#### Independent variables
* Device type: {CPU, GPU}
* Number of devices: {1~~,2~~}
* ~~Distribution: {single,copy,block,overlap}~~
* ~~Overlap size: NUMERICAL~~
* Stencil type: {MapOverlap,Stencil}.

#### Dependent variables
* Runtimes:
  * Total time
  * Per-skeleton:
    * ~~Data upload~~
    * ~~Data download~~
    * ~~Time spent executing job~~

#### Hardware
* Machine 1 (GPU):
  * Intel(R) Core(TM) i7-2600K CPU
  * 16GiB memory
  * GeForce GTX 690 (1536 CUDA cores @ 823 MHz)

* Machine 2 (CPU):
  * Intel(R) Core(TM) i5-4570 CPU
  * 8GiB memory
