# SkelCL Benchmarks

### Canny Edge Detection

An edge detection operator, which applies four steps:

1. Gaussian blur to smooth the image and reduce noise
1. Sobel filter edge detection.
1. Non-maximum suppression to remove spurious response to edge
   detection.
1. Threshold operation to produce the final result.

###### Cited in:
* S. Breuer, M. Steuwer, and S. Gorlatch, “Extending the SkelCL
  Skeleton Library for Stencil Computations on Multi-GPU Systems,”
  HiStencils 2014, pp. 23–30, 2014.
* S. Breuer, M. Steuwer, and S. Gorlatch, “High-Level Programming of
  Stencil Computations on Multi-GPU Systems Using the SkelCL Library,”
  HiStencils 2014, vol. 24, no. 3, pp. 23–30, 2014.

#### canny

* Runtime: 1.487s
* SLOC: 212 cpp, 61 cl
* Skeletons: {MapOverlap\<float(float)\>, MapOverlap\<float(float)\>, MapOverlap\<float(float)\>, MapOverlap\<float(float)\>}
* Output:
```
[==DeviceList.cpp:90   000.139s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.139s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.280s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.280s  INFO] Using 1 OpenCL device(s) in total
[=MapOverlapDef.h:177  001.435s  INFO] MapOverlap kernel started
[=MapOverlapDef.h:177  001.445s  INFO] MapOverlap kernel started
[=MapOverlapDef.h:177  001.448s  INFO] MapOverlap kernel started
[=MapOverlapDef.h:177  001.451s  INFO] MapOverlap kernel started
Total Init time: 0.000144000005
Total Creation time: 0.001152000041
Total Gauß time: 0.000010000000
Total Sobel time: 0.000003000000
Total NSM time: 0.000003000000
Total Threshold time: 0.000002000000
Total Total time: 0.001313999994
Total Total no init time: 0.001169999945
[======SkelCL.cpp:84   001.469s  INFO] SkelCL terminating. Freeing all resources.
```

#### cannyStencil

* Runtime: 1.635s
* SLOC: 199 cpp, 61 cl
* Skeletons: {Stencil\<float(float)\>}
* Output:
```
[==DeviceList.cpp:90   000.170s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.170s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.300s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.300s  INFO] Using 1 OpenCL device(s) in total
[====StencilDef.h:241  001.628s  INFO] workgroupSize: 4096
[====StencilDef.h:302  001.628s  INFO] Allocate: 2352 bytes local memory
[====StencilDef.h:328  001.628s  INFO] starting Stencil with 512x512 - 32x4
[====StencilDef.h:241  001.629s  INFO] workgroupSize: 4096
[====StencilDef.h:302  001.629s  INFO] Allocate: 816 bytes local memory
[====StencilDef.h:328  001.629s  INFO] starting Stencil with 512x512 - 32x4
[====StencilDef.h:241  001.630s  INFO] workgroupSize: 4096
[====StencilDef.h:302  001.630s  INFO] Allocate: 816 bytes local memory
[====StencilDef.h:328  001.630s  INFO] starting Stencil with 512x512 - 32x4
[====StencilDef.h:241  001.630s  INFO] workgroupSize: 8192
[====StencilDef.h:302  001.630s  INFO] Allocate: 512 bytes local memory
[====StencilDef.h:328  001.630s  INFO] starting Stencil with 512x512 - 32x4
Total Init time: 0.000132000001
Total Creation time: 0.001325999969
Total Total time: 0.001495999983
Total no init Total time: 0.001364000025
[======SkelCL.cpp:84   001.664s  INFO] SkelCL terminating. Freeing all resources.
[====Skeleton.cpp:79   001.664s  INFO] Event 0 timing: 34.1109ms
[====Skeleton.cpp:79   001.664s  INFO] Event 1 timing: 0.51329ms
[====Skeleton.cpp:79   001.664s  INFO] Event 2 timing: 0.51147ms
[====Skeleton.cpp:79   001.664s  INFO] Event 3 timing: 0.083103ms
```


### Dot Product

Standard dot product: `sum([i*j for i,j in A,B])`.

#### dot_product

* Runtime: 0.337s
* SLOC: 121 cpp
* Skeletons: {Zip\<int(int, int)\>, Reduce\<int(int)\>}
* Output:
```
[==DeviceList.cpp:90   000.001s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.001s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.129s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.129s  INFO] Using 1 OpenCL device(s) in total
[========main.cpp:109  000.329s  INFO] skelcl: 5023424
```


### Conway's Game of Life

#### gameoflife

* Runtime: 0.339s
* SLOC: 92 cpp 12 cl
* Skeletons: {MapOverlap\<int(int)\>}
* Output:
```
[==DeviceList.cpp:90   000.001s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.001s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.141s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.141s  INFO] Using 1 OpenCL device(s) in total
[=MapOverlapDef.h:177  000.375s  INFO] MapOverlap kernel started
Init time : 0.000140999997
Total time : 0.000376000011
Total without init time : 0.000235000000
[======SkelCL.cpp:84   000.376s  INFO] SkelCL terminating. Freeing all resources.
```

#### gameoflifeStencil

* Runtime: 0.380s
* SLOC: 98 cpp 12 cl
* Skeletons: {Stencil\<int(int)\>}
* Output:
```
[==DeviceList.cpp:90   000.001s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.001s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.123s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.123s  INFO] Using 1 OpenCL device(s) in total
[====StencilDef.h:241  000.348s  INFO] workgroupSize: 4096
[====StencilDef.h:302  000.348s  INFO] Allocate: 816 bytes local memory
[====StencilDef.h:328  000.348s  INFO] starting Stencil with 32x8 - 32x4
Init time : 0.000123999998
Input time : 0.000000000000
Creation time : 0.000222999995
Exec time : 0.000001000000
Total time : 0.000348000001
Total without init time : 0.000224000003
[======SkelCL.cpp:84   000.348s  INFO] SkelCL terminating. Freeing all resources.
[====Skeleton.cpp:79   000.348s  INFO] Event 0 timing: 0.050648ms
```


## Gaussian Blur

Apply Gaussian function to each pixel. The `gauss` and `gaussStencil`
use a 2D range, `gaussY` and `gaussYStencil` blue in only one
dimension.

###### Cited in:
* S. Breuer, M. Steuwer, and S. Gorlatch, “Extending the SkelCL
  Skeleton Library for Stencil Computations on Multi-GPU Systems,”
  HiStencils 2014, pp. 23–30, 2014.
* S. Breuer, M. Steuwer, and S. Gorlatch, “High-Level Programming of
  Stencil Computations on Multi-GPU Systems Using the SkelCL Library,”
  HiStencils 2014, vol. 24, no. 3, pp. 23–30, 2014.


#### gauss

* Runtime: 0.646s
* SLOC: 165 cpp 47 cl
* Skeletons: {MapOverlap\<int(int)\>}
* Output:
```
[==DeviceList.cpp:90   000.001s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.001s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.128s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.128s  INFO] Using 1 OpenCL device(s) in total
[=MapOverlapDef.h:177  000.612s  INFO] MapOverlap kernel started
```

#### gaussStencil

* Runtime: 1.091s
* SLOC: 210 cpp 16 cl
* Skeletons: {Stencil\<float(float)\>}
* Output:
```
[==DeviceList.cpp:90   000.149s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.149s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.273s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.273s  INFO] Using 1 OpenCL device(s) in total
[====StencilDef.h:241  000.992s  INFO] workgroupSize: 4096
[====StencilDef.h:302  000.992s  INFO] Allocate: 2352 bytes local memory
[====StencilDef.h:328  000.992s  INFO] starting Stencil with 512x512 - 32x4
Init time : 0.000125999999
Input time : 0.000000000000
Creation time : 0.000715999980
Exec time : 0.000003000000
Total time : 0.000844999973
Total without init time : 0.000719000003
[======SkelCL.cpp:84   001.014s  INFO] SkelCL terminating. Freeing all resources.
[====Skeleton.cpp:79   001.015s  INFO] Event 0 timing: 3.24317ms
```

#### gaussY

* Runtime: 0.780s
* SLOC: 200 cpp 12 cl
* Skeletons: {MapOverlap\<float(float)\>}
* Output:
```
[==DeviceList.cpp:90   000.148s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.148s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.275s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.275s  INFO] Using 1 OpenCL device(s) in total
[=MapOverlapDef.h:177  000.580s  INFO] MapOverlap kernel started
Init time : 0.000130000000
Creation time : 0.000302000000
Exec time all iter: 0.000003000000
Total time : 0.000434999994
Total without init time : 0.000304999994
[======SkelCL.cpp:84   000.715s  INFO] SkelCL terminating. Freeing all resources.
```

#### gaussYStencil

* Runtime: 0.910s
* SLOC: 202 cpp 12 cl
* Skeletons: {Stencil\<float(float)\>}
* Output:
```
[==DeviceList.cpp:90   000.145s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.145s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.284s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.284s  INFO] Using 1 OpenCL device(s) in total
[====StencilDef.h:241  000.827s  INFO] workgroupSize: 4096
[====StencilDef.h:302  000.827s  INFO] Allocate: 1792 bytes local memory
[====StencilDef.h:328  000.827s  INFO] starting Stencil with 512x512 - 32x4
Init time : 0.000140999997
Creation time : 0.000540999987
Exec time all iter: 0.000002000000
Total time : 0.000684999977
Total without init time : 0.000544000010
[======SkelCL.cpp:84   000.847s  INFO] SkelCL terminating. Freeing all resources.
[====Skeleton.cpp:79   000.847s  INFO] Event 0 timing: 0.584592ms
```

## Heat Simulation

#### heat

* Runtime: 0.437s
* SLOC: 180 cpp 13 cl
* Skeletons: {MapOverlap\<float(float)\>}
* Output:
```
[==DeviceList.cpp:90   000.029s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.029s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.152s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.152s  INFO] Using 1 OpenCL device(s) in total
[=MapOverlapDef.h:177  000.395s  INFO] MapOverlap kernel started
Total time : 0.000371000002
Total without init time : 0.000246000011
[======SkelCL.cpp:84   000.398s  INFO] SkelCL terminating. Freeing all resources.
```

#### heatStencil

* Runtime: 0.473s
* SLOC: 177 cpp 13 cl
* Skeletons: {Stencil\<float(float)\>}
* Output:
```
[==DeviceList.cpp:90   000.030s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.030s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.153s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.153s  INFO] Using 1 OpenCL device(s) in total
[====StencilDef.h:241  000.389s  INFO] workgroupSize: 4096
[====StencilDef.h:302  000.389s  INFO] Allocate: 816 bytes local memory
[====StencilDef.h:328  000.389s  INFO] starting Stencil with 1024x1024 - 32x4
Total time : 0.000364000007
Total without init time : 0.000237999993
[======SkelCL.cpp:84   000.392s  INFO] SkelCL terminating. Freeing all resources.
[====Skeleton.cpp:79   000.392s  INFO] Event 0 timing: 1.90437ms
```

### Mandelbrot Set

###### Cited in:
* M. Steuwer, P. Kegel, and S. Gorlatch, “SkelCL - A Portable Skeleton
  Library for High-Level GPU Programming,” in Parallel and Distributed
  Processing Workshops and Phd Forum (IPDPSW), 2011 IEEE International
  Symposium on, 2011, pp. 1176–1182.
* M. Steuwer and S. Gorlatch, “SkelCL: Enhancing OpenCL for High-Level
  Programming of Multi-GPU Systems,” Parallel Comput. Technol.,
  vol. 7979, pp. 258–272, 2013.

#### mandelbrot

Output struct represents a pixel using 3 `unsigned char` variables for
RGB components.

* Runtime: 1.342s
* SLOC: 133 cpp 78 cl
* Skeletons: {Map\<struct(IndexPoint)\>}
* Output:
```
[==DeviceList.cpp:90   000.001s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.001s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.133s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.133s  INFO] Using 1 OpenCL device(s) in total
```

## Matrix Multiply

###### Cited in:
* M. Steuwer, M. Friese, S. Albers, and S. Gorlatch, “Introducing and
  implementing the allpairs skeleton for programming multi-GPU
  Systems,” Int. J. Parallel Program., vol. 42, pp. 601–618, 2014.

#### matrix_mult

* Runtime: 1.231s
* SLOC: 267 cpp
* Skeletons: {AllPairs\<int(int)\>}
* Output:
```
[==DeviceList.cpp:90   000.002s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.002s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.141s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.141s  INFO] Using 1 OpenCL device(s) in total
[========main.cpp:87   000.141s  INFO] started: multiplication of matrices A (1024 x 1024) and B (1024 x 1024)
[===AllPairsDef.h:90   001.040s DEBUG] Create new AllPairs object (0xe56ce0)
[===AllPairsDef.h:173  001.096s DEBUG] dim: 1024 height: 1024 width: 1024
[===AllPairsDef.h:175  001.096s DEBUG] local: 32,8 global: 1024,64
[===AllPairsDef.h:209  001.102s  INFO] AllPairs kernel started
[========main.cpp:117  001.173s  INFO] finished: matrix C (1024 x 1024) calculated, elapsed time: 81 ms
[========main.cpp:263  001.176s  INFO] sizes: 1024, 1024, 1024; average time: 81 ms
```

## SAXPY

#### saxpy

* Runtime: 0.311s
* SLOC: 149 cpp
* Skeletons: {Zip\<float(float, float)\>}
* Output:
```
[==DeviceList.cpp:90   000.001s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.001s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.139s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.139s  INFO] Using 1 OpenCL device(s) in total
[========main.cpp:104  000.262s  INFO] Time: 5 ms
```

#### SkelFDTD

Also prints output to `log.log`.

* Runtime: 21.105s
* SLOC: 375 cpp 127 cl
* Skeletons: {Map\<struct(struct)\>, Stencil\<struct(struct)\>, Stencil\<struct(struct)\>}
* Output:
```
[==DeviceList.cpp:90   000.002s  INFO] 1 OpenCL platform(s) found
[==DeviceList.cpp:101  000.002s  INFO] 1 device(s) for OpenCL platform `Intel(R) OpenCL' found
[======Device.cpp:118  000.137s  INFO] Using device `Intel(R) Core(TM) i5-4570 CPU @ 3.20GHz' with id: 0
[==DeviceList.cpp:122  000.138s  INFO] Using 1 OpenCL device(s) in total
#define Pr 1.000000e+13f
#define eps_r 4.000000e+00f
#define eps_b 1.000000e+00f
#define abs_cell_size 32
#define array_size 2048
#define array_size_2 1024
#define log_2 6.931472e-01f
#define pi 3.141593e+00f
#define dt_r 1.667820e-17f
#define k 1.192011e-03f
#define a1 1.580421e+04f
#define a2 -1.530277e-03f
#define omega_a2 1.421223e+31f
#define Nges 3.311000e+24f
#define dt_tau10 1.667820e-05f
#define dt_tau32 1.667820e-04f
#define dt_tau21 1.667820e-07f
#define sqrt_eps0_mu0 2.654419e-03f
#define c 2.997924e+08f
#define src_x 50
#define src_y 50
#define idx_x (get_global_id(0))
#define idx_y (get_global_id(1))


data_t motionEquation(data_t N, data_t_matrix_t mE)
{
	if (N.w > 0.0f)
	{
		data_t E = get(mE, idx_y, idx_x);

		float N0 = (Nges - (N.x + N.y + N.z));
		N.x = (1.0f - dt_tau32)  * N.x + ((N0 < 0.0f) ? 0.0f : N0) * (N.w * dt_r);
		N.y = (1.0f - dt_tau21)  * N.y + N.x * dt_tau32 + a1 * (E.z * E.w);
		N.z = (1.0f - dt_tau10)  * N.z + N.y * dt_tau21 - a1 * (E.z * E.w);

		E.w = a2 * E.w - omega_a2 * E.x * dt_r + k * (N.z - N.y) * E.z * dt_r;

		E.x = E.x + E.w * dt_r;

		set(mE, idx_y, idx_x, E);
	}
	return N;
}
```
