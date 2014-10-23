# Import common build script.
include $(top_srcdir)/Makefile.common

noinst_PROGRAMS = 			\
	skel-n-queens			\
	$(NULL)

skel_n_queens_SOURCES = skel-n-queens.cc

#------------------------------------------------------------------------
# Auxiliary build targets.
#------------------------------------------------------------------------

# Execute all binaries.
run-local: $(noinst_PROGRAMS)
	@for bin in $^; do			\
		echo ./$$bin;			\
		./$$bin;			\
	done
