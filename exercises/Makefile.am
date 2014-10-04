SUBDIRS = templates

if HAVE_MPI_TOOLS
SUBDIRS += mpi
endif

noinst_PROGRAMS = 			\
	skel-n-queens			\
	$(NULL)

skel_n_queens_SOURCES = skel-n-queens.cc

# Static analysis:
if HAVE_CLANG
analyze_cxx_SRC = $(wildcard *.cc)
analyze_cxx_PLISTS = $(analyze_cxx_SRC:%.cc=%.plist)
MOSTLYCLEANFILES = $(analyze_cxx_PLISTS)
.PHONY: analyze
all-local: analyze
analyze: analyze_cxx
analyze_cxx: $(analyze_cxx_PLISTS)
$(analyze_cxx_PLISTS): %.plist: %.cc
	@echo '  CXXSA    $@'
	@$(CXXCOMPILE) --analyze $< -o $@
endif
