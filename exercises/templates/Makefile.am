BIN := $(basename $(wildcard *.cc))

CLEANFILES = $(BIN:=.out)

GCC_FLAGS = -Wall -Wextra -Wzero-as-null-pointer-constant -Wno-unused-parameter

# Compilation target for C++ programs:
%.out: %.cc %.h
	@echo '  G++    $@'
	@g++ $(GCC_FLAGS) $< -o $@

all: $(BIN:=.out)

# Execute all programs:
run: all
	@for f in $(BIN:=.out); do \
		echo ./$$f; \
		./$$f; \
	done
