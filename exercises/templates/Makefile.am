BIN := $(basename $(wildcard *.cc))

CLEANFILES = $(BIN:=.out)

# Compilation target for C++ programs:
%: %.cc
	g++ $< -o $@.out

all: $(BIN)

# Execute all programs:
run: all
	@for f in $(BIN:=.out); do \
		echo ./$$f; \
		./$$f; \
	done
