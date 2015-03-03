# E1: Characterisation of SkelCL performance

The goal is to collect data about SkelCL performance under various
conditions (independent variables) which will be used to identify an
*optimisation space*.

#### Benchmarks

###### Real world programs
  * Canny edge detection
  * FDTD
  * Game of Life
  * Gaussian blur
  * Heat simulation
  * Mandelbrot set
  * Sobel edge detection

###### Standard algorithms
  * Dot product
  * Matrix multiply
  * SAXPY

#### Independent variables
* Device type: {CPU, GPU}
* Number of devices: {1}
* Distribution: {single,copy,block,overlap}
* Overlap size: NUMERICAL
* Device properties: number of cores, amount of memory, memory
  bandwidth.
* Program: (enumerated list of all benchmarks).
* Skeleton: {Map,Reduce,Scan,Zip,MapOverlap,AllPairs,Stencil}.
* Runtime properties: size of input data.
* OpenCL kernel properties: size of user functions, number of
  branches, number of loads/stores.

#### Dependent variables
* Runtimes (for each skeleton call):
  * Total time
  * Data upload
  * Data download
  * Time spent executing job

#### Hardware
* Intel(R) Core(TM) i7-2600K CPU
* 16GiB memory
* GeForce GTX 690 (1536 CUDA cores @ 823 MHz)
