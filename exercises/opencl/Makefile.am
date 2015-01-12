# Import common build script.
include $(top_srcdir)/Makefile.common

###############################################################################
# Build targets and configuration.
###############################################################################

CL_INCLUDE = $(CL)/include
CL_LIB = $(CL)/lib64

# Local compiler and linker flags.
AM_CXXFLAGS = -isystem $(CL_INCLUDE)
AM_LDFLAGS  = -L$(CL_LIB) -lOpenCL

#------------------------------------------------------------------------
# Build targets.
#------------------------------------------------------------------------
noinst_PROGRAMS = 			\
	hello				\
	info				\
	$(NULL)

#------------------------------------------------------------------------
# Build target requisites.
#------------------------------------------------------------------------
hello_SOURCES = hello.cc
info_SOURCES = info.cc

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin; 			\
	done
