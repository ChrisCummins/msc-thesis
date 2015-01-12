# Import common build script.
include $(top_srcdir)/Makefile.common

SUBDIRS = 				\
	algos				\
	llvm				\
	opencl				\
	tbb				\
	templates			\
	$(NULL)

if HAVE_MPI_TOOLS
SUBDIRS += mpi
endif
