# Import common build script.
include $(top_srcdir)/Makefile.common

BIN := $(basename $(wildcard *.cpp))
NO_OF_PROCESSES := 4

CLEANFILES = $(BIN:=.out)

# Compilation target for C++ programs:
%: %.cpp
	mpic++ $< -o $@.out

all: $(BIN)

# Execute all programs:
run-local: all
	@for f in $(BIN:=.out); do \
		echo mpirun -np $(NO_OF_PROCESSES) $$f; \
		mpirun -np $(NO_OF_PROCESSES) $$f; \
	done
