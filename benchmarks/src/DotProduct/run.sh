#!/bin/sh

. ../include/testlib.sh

# Computation of the dot product of two randomly created vectors.
#
# 	[--help] (default value: true)
# 		Prints this help message and then exits the program.
#
# 	[--device_count] (default value: 1)
# 		Number of devices used by SkelCL.
#
# 	[--device_type] (default value: ANY)
# 		Device type: ANY, CPU, GPU, ACCELERATOR
#
# 	[-l --logging --verbose_logging] (default value: false)
# 		Enable verbose logging.
#
# 	[-n --size] (default value: 1024)
# 		Size of the two vectors used in the computation.
#
# 	[-c --check --check_result] (default value: true)
# 		Check parallel computed result against a sequential computed version.
bin=benchmark
check_result=1
device_count=1
device_type=ANY
logging=0


size=5000000

append --size $size
