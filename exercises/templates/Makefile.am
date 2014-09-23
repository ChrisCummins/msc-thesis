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

noinst_PROGRAMS = 			\
	dac-float			\
	dac-int				\
	dc-merge-sort			\
	fddc-merge-sort			\
	merge-sort			\
	stack				\
	std-stable-sort-list-int	\
	std-stable-sort-vector-int	\
	$(NULL)

dac_float_SOURCES = dac-float.cc list.cc timer.cc
dac_int_SOURCES = dac-int.cc list.cc timer.cc
dc_merge_sort_SOURCES = dc-merge-sort.cc list.cc timer.cc
fddc_merge_sort_SOURCES = fddc-merge-sort.cc list.cc timer.cc
merge_sort_SOURCES = merge-sort.cc list.cc timer.cc
stack_SOURCES = stack.cc
std_stable_sort_list_int_SOURCES = std-stable-sort-list-int.cc list.cc timer.cc
std_stable_sort_vector_int_SOURCES = std-stable-sort-vector-int.cc timer.cc

# Execute all programs:
run: all
	@for f in $(noinst_PROGRAMS); do \
		echo ./$$f; \
		./$$f; \
	done
