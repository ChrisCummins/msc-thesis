AM_CXXFLAGS = 				\
	-std=c++11			\
	-pthread 			\
	-Wall 				\
	-Wextra 			\
	-Wno-unused-parameter 		\
	-O0 -pg 			\
	-DDAC_DEBUG_LEVEL=1		\
	$(NULL)

AM_LDFLAGS = 				\
	-pthread			\
	$(NULL)

noinst_PROGRAMS = 			\
	ita-merge-sort-int		\
	merge-sort-float		\
	merge-sort-int			\
	std-stable-sort-int		\
	$(NULL)

test_SOURCES = timer.cc test.cc

ita_merge_sort_int_SOURCES      = ita-merge-sort-int.cc $(test_SOURCES)
merge_sort_float_SOURCES	= merge-sort-float.cc $(test_SOURCES)
merge_sort_int_SOURCES		= merge-sort-int.cc $(test_SOURCES)
std_stable_sort_int_SOURCES	= std-stable-sort-int.cc $(test_SOURCES)

# Execute all programs:
run: all
	@for f in $(noinst_PROGRAMS); do \
		echo ./$$f; \
		./$$f; \
	done
