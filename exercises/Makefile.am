# Import common build script.
include $(top_srcdir)/Makefile.common

SUBDIRS = 				\
	llvm				\
	templates			\
	$(NULL)

if HAVE_MPI_TOOLS
SUBDIRS += mpi
endif

noinst_PROGRAMS = 			\
	skel-n-queens			\
	$(NULL)

skel_n_queens_SOURCES = skel-n-queens.cc
