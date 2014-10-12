# Import common build script.
include $(top_srcdir)/Makefile.common

###############################################################################
# Build targets and configuration.
###############################################################################

# Local compiler and linker flags.
AM_CXXFLAGS = $(shell mpic++ --showme:compile)
AM_LDFLAGS  = $(shell mpic++ --showme:link)

#------------------------------------------------------------------------
# Build targets.
#------------------------------------------------------------------------
noinst_PROGRAMS = 			\
	hello				\
	pi				\
	$(NULL)

#------------------------------------------------------------------------
# Define build target requisites.
#------------------------------------------------------------------------
hello_SOURCES  = hello.cpp
pi_SOURCES     = pi.cpp

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for f in $^; do \
		echo "mpirun -np $(NUM_PROC) $$f"; \
		mpirun -np $(NUM_PROC) $$f; \
	done
