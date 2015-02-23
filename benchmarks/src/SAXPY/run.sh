#!/bin/sh

. ../include/testlib.sh

# Computation of 'Single-Precision A.X + Y'.
#
# 	[--help] (default value: true)
# 		Prints this help message and then exits the program.
#
# 	[--device_count] (default value: 2)
# 		Number of devices used by SkelCL.
#
# 	[--device_type] (default value: ANY)
# 		Device type: ANY, CPU, GPU, ACCELERATOR
#
# 	[-l --logging --verbose_logging] (default value: false)
# 		Enable verbose logging.
#
# 	[-n --size] (default value: 1048576)
# 		Size of the two vectors used in the computation.
#
# 	[-c --check --check_result] (default value: false)
# 		Check SkelCL computation against sequential computed version.
bin=benchmark
device_count=1
device_type=ANY
logging=0


check_result=1
size=10000000

append --size $size
