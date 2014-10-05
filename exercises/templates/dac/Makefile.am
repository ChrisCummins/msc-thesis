AM_CXXFLAGS = -pthread
AM_LDFLAGS  = -pthread

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
run-local: all
	@for f in $(noinst_PROGRAMS); do \
		echo ./$$f; \
		./$$f; \
	done


# The C++ static analyser.
##########################
#
# If we have clang++ available, then we can invoke it with the
# --analyze parameter to run a static analyser.
#
if HAVE_CLANG

analyze_SRC = $(wildcard *.h)
analyze_PLISTS = $(addsuffix .plist,$(analyze_SRC))
MOSTLYCLEANFILES += $(analyze_PLISTS) $(addsuffix .plist,$(sort $(SOURCES)))

# Our static analysis function:
define analyze
	$(AM_V_at)$(CXXCOMPILE) -xc++ --analyze $1 -o $2;
endef

all-local: $(analyze_PLISTS)

# Explicit target for static analysis:
$(analyze_PLISTS): %.plist: %
	@if ! $(AM_V_P); then echo '  CXXSA    $<'; fi
	$(call analyze,$<,$@)

else
# If we don't have clang++ then just define a null function so that we
# can safely call clang++ from our non-conditional make rule:
define analyze
endef
endif


# The C++ linter.
#################
#
# The cpplint script checks an input source file and enforces the
# style guidelines set out in:
#
#   http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
#
if HAVE_CPPLINT

cpplint_SRC       = $(wildcard *.h)
cpplint_LINT      = $(addsuffix .lint,$(cpplint_SRC))
MOSTLYCLEANFILES += $(cpplint_LINT) $(addsuffix .lint,$(sort $(SOURCES)))

# Our cpplint function:
define cpplint
	$(AM_V_at)$(CPPLINT) --filter=-legal,-build/c++11 $1 2>&1 \
		| grep -v '^Done processing\|^Total errors found: ' \
		| tee $2
endef

all-local: $(cpplint_LINT)

# Explicit target for cpplint:
$(cpplint_LINT): %.lint: %
	@if ! $(AM_V_P); then echo '  CPPLINT  $<'; fi
	$(call cpplint,$<,$@)

else
# If we don't have cpplint then just define a null function so that we
# can safely call cpplint from our non-conditional make rule:
define cpplint
endef
endif


# C++ compilation.
##################
#
# Automake doesn't support extending the built-in rules, so we have to
# override the default with our own. The first two lines are copied
# from the built-in the rule. We then hook our custom build logic at
# the end, running the static analyzer and lint script.
%.o: %.cc
	$(AM_V_CXX)$(CXXCOMPILE) -MT $@ -MD -MP -MF $(DEPDIR)/$*.Tpo -c -o $@ $<
	$(AM_V_at)$(am__mv) $(DEPDIR)/$*.Tpo $(DEPDIR)/$*.Po
	$(call analyze,$<,$<.plist)
	$(call cpplint,$<,$<.lint)
