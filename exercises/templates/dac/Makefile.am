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
	ita-merge-sort-int			\
	merge-sort-float			\
	merge-sort-int				\
	skel-dac-merge-sort-int			\
	skel-merge-sort-int			\
	skel-max-subarray			\
	std-stable-sort-int			\
	$(NULL)

#------------------------------------------------------------------------
# Define build target requisites.
#------------------------------------------------------------------------
TestSources = timer.cc test.cc

ita_merge_sort_int_SOURCES      = ita-merge-sort-int.cc      $(TestSources)
merge_sort_float_SOURCES        = merge-sort-float.cc        $(TestSources)
merge_sort_int_SOURCES          = merge-sort-int.cc          $(TestSources)
skel_dac_merge_sort_int_SOURCES = skel-dac-merge-sort-int.cc $(TestSources)
skel_merge_sort_int_SOURCES     = skel-merge-sort-int.cc     $(TestSources)
skel_max_subarray_SOURCES       = skel-max-subarray.cc       $(TestSources)
std_stable_sort_int_SOURCES     = std-stable-sort-int.cc     $(TestSources)

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin;			\
	done
