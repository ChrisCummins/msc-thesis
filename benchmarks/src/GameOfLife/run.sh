#!/bin/sh

. ../include/testlib.sh

# Computation of the Gaussian blur.

# 	[--help] (default value: true)
# 		Prints this help message and then exits the program.
#
# 	[--device_count] (default value: 1)
# 		Number of devices used by SkelCL.
#
# 	[--device_type] (default value: ANY)
# 		Device type: ANY, CPU, GPU, ACCELERATOR
#
# 	[-i --iterations] (default value: 1)
# 		The number of iterations
#
# 	[-n --problem_size] (default value: 5)
# 		Number of devices used by SkelCL.
bin=benchmark
device_count=1
device_type=ANY
logging=0

iterations=1000
problem_size=500

append --iterations $iterations
append --problem_size $problem_size
