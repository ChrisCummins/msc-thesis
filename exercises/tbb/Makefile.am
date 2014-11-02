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
	game_of_life				\
	quicksort 				\
	$(NULL)

#------------------------------------------------------------------------
# Define build target requisites.
#------------------------------------------------------------------------
quicksort_SOURCES               = quicksort.cc
game_of_life_SOURCES            = Evolution.cc Update_state.cc game_of_life.cc

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin;			\
	done
