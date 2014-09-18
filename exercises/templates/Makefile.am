BIN := $(basename $(wildcard *.cpp))

CLEANFILES = $(BIN:=.out)

# Compilation target for C++ programs:
%: %.cpp
	g++ $< -o $@.out

all: $(BIN)

# Execute all programs:
run: all
	@for f in $(BIN:=.out); do \
		echo ./$$f; \
		./$$f; \
	done
