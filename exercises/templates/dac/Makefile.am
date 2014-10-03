CXX_STD = -std=c++11

if HAVE_CLANG
CXX_STDLIB = -stdlib=libstdc++
all-local: analyze
else
CXX_STDLIB =
endif

AM_CXXFLAGS = 				\
	$(CXX_STD)			\
	$(CXX_STDLIB)			\
	-pthread 			\
	-Wall 				\
	-Wextra 			\
	-Wno-unused-parameter 		\
	-O0 -pg 			\
	-DDAC_DEBUG_LEVEL=1		\
	$(NULL)

AM_LDFLAGS = 				\
	$(CXX_STD)			\
	$(CXX_STDLIB)			\
	-pthread			\
	$(NULL)

noinst_PROGRAMS = 			\
	ita-merge-sort-int		\
	merge-sort-float		\
	merge-sort-int			\
	skel-merge-sort-int		\
	std-stable-sort-int		\
	$(NULL)

test_SOURCES = timer.cc test.cc

ita_merge_sort_int_SOURCES      = ita-merge-sort-int.cc $(test_SOURCES)
merge_sort_float_SOURCES	= merge-sort-float.cc $(test_SOURCES)
merge_sort_int_SOURCES		= merge-sort-int.cc $(test_SOURCES)
skel_merge_sort_int_SOURCES     = skel-merge-sort-int.cc $(test_SOURCES)
std_stable_sort_int_SOURCES	= std-stable-sort-int.cc $(test_SOURCES)

# Execute all programs:
run: all
	@for f in $(noinst_PROGRAMS); do \
		echo ./$$f; \
		./$$f; \
	done

# Static analysis:
analyze_cxx_SRC = $(wildcard *.cc)
analyze_cxx_PLISTS = $(analyze_cxx_SRC:%.cc=%.plist)

MOSTLYCLEANFILES = $(analyze_cxx_PLISTS)

.PHONY: analyze
analyze: analyze_cxx

analyze_cxx: $(analyze_cxx_PLISTS)

$(analyze_cxx_PLISTS): %.plist: %.cc
	@echo '  CXXSA    $@'
	@$(CXXCOMPILE) --analyze $< -o $@
