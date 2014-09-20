AM_CFLAGS = -Wall -Wextra -Wno-unused-parameter -Wstrict-prototypes -O0 -g

noinst_PROGRAMS = stack dac
stack_SOURCES   = stack.cc
dac_SOURCES     = dac.cc list.cc timer.cc

# Execute all programs:
run: all
	@for f in $(noinst_PROGRAMS); do \
		echo ./$$f; \
		./$$f; \
	done
