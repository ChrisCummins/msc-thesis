PROGRAMS := $(basename $(wildcard *.cpp))

%: %.cpp
	mpic++ $< -o $@.out

all: $(PROGRAMS)

run: all
	@for f in $(PROGRAMS:=.out); do \
		echo mpirun -np 4 $$f; \
		mpirun -np 4 $$f; \
	done

clean:
	rm -fv $(PROGRAMS:=.out)
