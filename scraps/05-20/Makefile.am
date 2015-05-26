# Import common build script.
include $(top_srcdir)/Makefile.common

###############################################################################
# Build targets and configuration.
###############################################################################

# Local compiler and linker flags.
AM_CXXFLAGS = \
	-I $(SKELCL_DIR)/include \
	-I $(SKELCL_DIR)/libraries/pvsutil/include \
	-I $(SKELCL_DIR)/libraries/stooling/include
LDADD       = \
	-L$(SKELCL_BUILD_DIR)/src \
	-L$(SKELCL_BUILD_DIR)/libraries/pvsutil/src \
	-L$(SKELCL_BUILD_DIR)/libraries/stooling/src \
	-L$(CL)/lib64 \
	-lOpenCL -lSkelCL -lpvsutil -lstooling

#------------------------------------------------------------------------
# Build targets.
#------------------------------------------------------------------------
noinst_PROGRAMS = 			\
	mandelbrot_opencl		\
	mandelbrot_seq			\
	mandelbrot_skelcl		\
	$(NULL)

#------------------------------------------------------------------------
# Build target requisites.
#------------------------------------------------------------------------
mandelbrot_opencl_SOURCES = mandelbrot_opencl.cc common.cc
mandelbrot_seq_SOURCES = mandelbrot_seq.cc common.cc
mandelbrot_skelcl_SOURCES = mandelbrot_skelcl.cc common.cc

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin; 			\
	done
