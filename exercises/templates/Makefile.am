AM_CXXFLAGS = 				\
	-std=c++11			\
	-pthread 			\
	-Wall 				\
	-Wextra 			\
	-Wno-unused-parameter 		\
	-O0 -pg 			\
	$(NULL)

AM_LDFLAGS = 				\
	-pthread			\
	$(NULL)

noinst_PROGRAMS = stack dac dc-merge-sort fddc-merge-sort lib-sort merge-sort
stack_SOURCES = stack.cc
dac_SOURCES = dac.cc list.cc timer.cc
dc_merge_sort_SOURCES = dc-merge-sort.cc list.cc timer.cc
fddc_merge_sort_SOURCES = fddc-merge-sort.cc list.cc timer.cc
lib_sort_SOURCES = lib-sort.cc list.cc timer.cc
merge_sort_SOURCES = merge-sort.cc list.cc timer.cc

# Execute all programs:
run: all
	@for f in $(noinst_PROGRAMS); do \
		echo ./$$f; \
		./$$f; \
	done
