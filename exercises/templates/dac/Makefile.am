###############################################################################
# Build targets and configuration.
###############################################################################

# Local compiler and linker flags.
AM_CXXFLAGS = -pthread
AM_LDFLAGS  = -pthread

MOSTLYCLEANFILES =

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
Headers         = $(wildcard *.h)
Sources         = $(sort $(SOURCES))
SourceBaseNames = $(patsubst %.cc,%,$(Sources))
TestSources     = timer.cc test.cc

ita_merge_sort_int_SOURCES      = ita-merge-sort-int.cc      $(TestSources)
merge_sort_float_SOURCES        = merge-sort-float.cc        $(TestSources)
merge_sort_int_SOURCES          = merge-sort-int.cc          $(TestSources)
skel_dac_merge_sort_int_SOURCES = skel-dac-merge-sort-int.cc $(TestSources)
skel_merge_sort_int_SOURCES     = skel-merge-sort-int.cc     $(TestSources)
skel_max_subarray_SOURCES       = skel-max-subarray.cc       $(TestSources)
std_stable_sort_int_SOURCES     = std-stable-sort-int.cc     $(TestSources)

###############################################################################
# Local compilation rules.
###############################################################################

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@echo "SOURCES: $(sort $(SOURCES))"
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin;			\
	done

#------------------------------------------------------------------------
# Static analysis of C++ files.
#------------------------------------------------------------------------

# If we have clang++ available, then we can invoke it with the
# --analyze parameter to perform static analysis of sources.
if HAVE_CLANG

# The clang static analyser generates .plist files.
StaticAnalysisPlists = $(addsuffix .plist,$(Sources) $(Headers))
MOSTLYCLEANFILES += $(StaticAnalysisPlists)

# Generate static analysis files as part of "all" target.
all-local: $(StaticAnalysisPlists)

# Explicit target for static analysis.
$(StaticAnalysisPlists): %.plist: %
	@if ! $(AM_V_P); then echo '  CXXSA    $<'; fi
	$(AM_V_at)$(call analyze,$<,$@)

# Function for generating plist files.
define analyze
	$(CXXCOMPILE) -xc++ --analyze $1 -o $2;
endef

else
# If we don't have clang++ then just define an empty function so that
# we can safely call analyze whether or not we are building with
# clang++.
define analyze
	true
endef
endif

#------------------------------------------------------------------------
# Style checking C++ files.
#------------------------------------------------------------------------

# The cpplint script checks an input source file and enforces the
# style guidelines set out in:
#
#   http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
#
if HAVE_CPPLINT

CppLintFiles      = $(addsuffix .lint,$(Sources) $(Headers))
MOSTLYCLEANFILES += $(CppLintFiles)

all-local: $(CppLintFiles)

# Explicit target for cpplint:
$(CppLintFiles): %.lint: %
	@if ! $(AM_V_P); then echo '  CPPLINT  $<'; fi
	$(AM_V_at)$(call cpplint,$<,$@)

# Function for generating lint files.
define cpplint
	$(CPPLINT) --filter=-legal,-build/c++11 $1 2>&1 		\
		| grep -v '^Done processing\|^Total errors found: ' 	\
		| tee $2
endef

else
# If we don't have cpplint then just define an empty function so that
# we can safely call cpplint whether or not we have the cpplint
# package available.
define cpplint
	true
endef
endif


#------------------------------------------------------------------------
# Generating LLVM IR bytecode.
#------------------------------------------------------------------------

# If we have clang++ available, then we emit LLVM bytecode in addition
# to the native object file. This bytecode isn't used by any other
# stage of the compilation process, but is intended primarily for
# informative purposes and to aid in debugging.
if HAVE_CLANG

# Generate bytecode files with .ll extension.
ByteCodeFiles     = $(addsuffix .ll,$(SourceBaseNames))
MOSTLYCLEANFILES += $(ByteCodeFiles)

# Function for generating bytecode.
define emit-bytecode
	$(CXXCOMPILE) -emit-llvm -S -o $2 $1
endef

else
# If we don't have clang++ then just define an empty function so that
# we can safely call emit-bytecode whether or not we are building with
# clang++.
define emit-bytecode
	true
endef
endif


#------------------------------------------------------------------------
# Generating native assembly sources.
#------------------------------------------------------------------------

# We generate native assembly code for all input sources. This is for
# debugging and informative purposes, it is not required as part of
# the compilation process.
AsmFiles          = $(addsuffix .s,$(SourceBaseName))
MOSTLYCLEANFILES += $(AsmFiles)

# Function for generating asm files.
define emit-asm
	$(CXX) -S -o $2 $1
endef


#------------------------------------------------------------------------
# Compiling C++ object files.
#------------------------------------------------------------------------

# Automake doesn't support extending the built-in rules, so we have to
# override the default with our own. The first two lines are copied
# from the built-in the rule. We then hook our custom build logic
# after this, generating additional source listings and performing
# source analysis.
%.o: %.cc
	$(AM_V_CXX)$(CXXCOMPILE) -MT $@ -MD -MP -MF $(DEPDIR)/$*.Tpo -c -o $@ $<
	$(AM_V_at)$(am__mv) $(DEPDIR)/$*.Tpo $(DEPDIR)/$*.Po
	$(AM_V_at)$(call emit-bytecode,$<,$*.ll)
	$(AM_V_at)$(call emit-asm,$*.ll,$*.s)
	$(AM_V_at)$(call analyze,$<,$<.plist)
	$(AM_V_at)$(call cpplint,$<,$<.lint)
