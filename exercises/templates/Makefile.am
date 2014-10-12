# Import common build script.
include $(top_srcdir)/Makefile.common

SUBDIRS = dac

noinst_PROGRAMS = 			\
	dc-merge-sort			\
	fddc-merge-sort			\
	merge-sort			\
	stack				\
	std-stable-sort-list-int	\
	$(NULL)

dc_merge_sort_SOURCES = dc-merge-sort.cc list.cc timer.cc
fddc_merge_sort_SOURCES = fddc-merge-sort.cc list.cc timer.cc
merge_sort_SOURCES = merge-sort.cc list.cc timer.cc
stack_SOURCES = stack.cc
std_stable_sort_list_int_SOURCES = std-stable-sort-list-int.cc list.cc timer.cc

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin;			\
	done
