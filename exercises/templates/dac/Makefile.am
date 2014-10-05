AM_CXXFLAGS = -pthread
AM_LDFLAGS = -pthread

noinst_PROGRAMS = 			\
	ita-merge-sort-int		\
	merge-sort-float		\
	merge-sort-int			\
	skel-dac-merge-sort-int		\
	skel-merge-sort-int		\
	skel-max-subarray		\
	std-stable-sort-int		\
	$(NULL)

test_SOURCES = timer.cc test.cc

ita_merge_sort_int_SOURCES      = ita-merge-sort-int.cc $(test_SOURCES)
merge_sort_float_SOURCES        = merge-sort-float.cc $(test_SOURCES)
merge_sort_int_SOURCES          = merge-sort-int.cc $(test_SOURCES)
skel_dac_merge_sort_int_SOURCES = skel-dac-merge-sort-int.cc $(test_SOURCES)
skel_merge_sort_int_SOURCES     = skel-merge-sort-int.cc $(test_SOURCES)
skel_max_subarray_SOURCES       = skel-max-subarray.cc $(test_SOURCES)
std_stable_sort_int_SOURCES     = std-stable-sort-int.cc $(test_SOURCES)

MOSTLYCLEANFILES =

# Execute all programs:
.PHONY: run
run: all
	@for f in $(noinst_PROGRAMS); do \
		echo ./$$f; \
		./$$f; \
	done


# Static analysis:
if HAVE_CLANG

all-local: analyze
analyze_SRC = $(wildcard *.h) $(wildcard *.cc)
analyze_PLISTS = $(addsuffix .plist,$(analyze_SRC))
MOSTLYCLEANFILES += $(analyze_PLISTS)

analyze: $(analyze_PLISTS)

$(analyze_PLISTS): %.plist: %
	@echo '  CXXSA    $<'
	@$(CXXCOMPILE) -xc++ --analyze $< -o $@
endif


# C++ linter:
if HAVE_CPPLINT

all-local: cpplint
cpplint_SRC = $(wildcard *.cc) $(wildcard *.h)
cpplint_LINT = $(addsuffix .lint,$(cpplint_SRC))
MOSTLYCLEANFILES += $(cpplint_LINT)

cpplint: $(cpplint_LINT)

$(cpplint_LINT): %.lint: %
	@echo '  CPPLINT  $<'
	@$(CPPLINT) --filter=-legal,-build/c++11 $< 2>&1 \
		| grep -v '^Done processing\|^Total errors found: ' \
		| tee $@
endif
