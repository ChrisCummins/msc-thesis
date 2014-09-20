AM_CFLAGS = -Wall -Wextra -Wno-unused-parameter -Wstrict-prototypes -O0 -g

noinst_PROGRAMS    = stack merge-sort
stack_SOURCES      = stack.cc
merge_sort_SOURCES = merge-sort.cc list.cc timer.cc

# Execute all programs:
run: all
	@for f in $(noinst_PROGRAMS); do \
		echo ./$$f; \
		./$$f; \
	done
