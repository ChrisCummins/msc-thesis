BIN := $(basename $(wildcard *.cc))

CLEANFILES = $(BIN:=.out)

GCC_FLAGS = -Wall -Wextra -Wzero-as-null-pointer-constant -Wno-unused-parameter

# Compilation target for C++ programs:
%: %.cc
	g++ $(GCC_FLAGS) $< -o $@.out

all: $(BIN)

# Execute all programs:
run: all
	@for f in $(BIN:=.out); do \
		echo ./$$f; \
		./$$f; \
	done
