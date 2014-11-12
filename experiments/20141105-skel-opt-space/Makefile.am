# Import common build script.
include $(top_srcdir)/Makefile.common

###############################################################################
# Build targets and configuration.
###############################################################################

# Local compiler and linker flags.
AM_CXXFLAGS = -pthread
AM_LDFLAGS  = -pthread

#------------------------------------------------------------------------
# Build targets.
#------------------------------------------------------------------------
noinst_PROGRAMS = 				\
	char					\
	$(NULL)

#------------------------------------------------------------------------
# Define build target requisites.
#------------------------------------------------------------------------
TestSources = test.cc

char_SOURCES = char.cc $(TestSources)

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin;			\
	done
