# Import common build script.
include $(top_srcdir)/Makefile.common

###############################################################################
# Build targets and configuration.
###############################################################################

# Local compiler and linker flags.
AM_CXXFLAGS =
AM_LDFLAGS  = -ltbb_debug

#------------------------------------------------------------------------
# Build targets.
#------------------------------------------------------------------------
noinst_PROGRAMS = 				\
	stable_sort				\
	$(NULL)

#------------------------------------------------------------------------
# Define build target requisites.
#------------------------------------------------------------------------
stable_sort_SOURCES = stable_sort.cc

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin;			\
	done
