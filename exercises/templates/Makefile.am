AM_CXXFLAGS = 				\
	-std=c++11			\
	-Wall 				\
	-Wextra 			\
	-Wno-unused-parameter 		\
	-O0 -g 				\
	$(NULL)

noinst_PROGRAMS    = stack dc-merge-sort fddc-merge-sort
stack_SOURCES      = stack.cc
dc_merge_sort_SOURCES = dc-merge-sort.cc list.cc timer.cc
fddc_merge_sort_SOURCES = fddc-merge-sort.cc list.cc timer.cc

# Execute all programs:
run: all
	@for f in $(noinst_PROGRAMS); do \
		echo ./$$f; \
		./$$f; \
	done
