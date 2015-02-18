# Import common build script.
include $(top_srcdir)/Makefile.common

###############################################################################
# Build targets and configuration.
###############################################################################

# Local compiler and linker flags.
AM_CXXFLAGS = -I$(CL)/include -I../include
AM_LDFLAGS  = -L$(CL)/lib64 -lOpenCL

#------------------------------------------------------------------------
# Build targets.
#------------------------------------------------------------------------
#
# TODO: The example program "mapoverlap" was causing the build to fail
#       with error:
#
#     mapoverlap.cpp:12:
#     In file included from ../include/skepu/mapoverlap.h:336:
#     ../include/skepu/src/mapoverlap.inl:196:27: error: invalid operands to binary expression
#          ('skepu::Vector<float>' and 'skepu::Vector<float>')
#       size_t size = inputEnd - inputBegin;
#                     ~~~~~~~~ ^ ~~~~~~~~~~
#
# Once this issue has been addressed, I should add "mapoverlap" to the
# list of noinst_PROGRAMS, and declare a built target requisite:
#
#     mapoverlap_SOURCES = mapoverlap.cpp
#
noinst_PROGRAMS = 			\
	map_test			\
	mapreduce			\
	generate			\
	2dmapoverlap			\
	reduce				\
	maparray			\
	scan				\
	$(NULL)

#------------------------------------------------------------------------
# Build target requisites.
#------------------------------------------------------------------------
map_test_SOURCES = map.cpp
mapreduce_SOURCES = mapreduce.cpp
generate_SOURCES = generate.cpp
2dmapoverlap_SOURCES = 2dmapoverlap.cpp
reduce_SOURCES = reduce.cpp
maparray_SOURCES = maparray.cpp
scan_SOURCES = scan.cpp

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin; 			\
	done
