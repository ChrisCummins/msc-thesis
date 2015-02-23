#!/bin/sh

. ../include/testlib.sh

# Computation of the Gaussian blur.
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
# 	[--range] (default value: 5)
# 		The Overlap radius
#
# 	[--inFile] (default value: lena.pgm)
# 		Filename of the input file
bin=benchmark
device_count=1
device_type=ANY

range=5
input_file=lena.pgm

append --range $range
append --inFile $input_file
