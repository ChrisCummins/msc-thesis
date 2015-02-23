#!/bin/sh

. ../include/testlib.sh

# Computation of the mandelbrot set.

# 	[--help] (default value: true)
# 		Prints this help message and then exits the program.

# 	[--device_count] (default value: 2)
# 		Number of devices used by SkelCL.

# 	[--device_type] (default value: ANY)
# 		Device type: ANY, CPU, GPU, ACCELERATOR

# 	[-l --logging --verbose_logging] (default value: false)
# 		Enable verbose logging.
bin=benchmark
device_count=1
device_type=ANY
logging=0
