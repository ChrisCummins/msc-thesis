all: cdecl computer_dating

cdecl: cdecl.o
computer_dating: computer_dating.o

%.o: %.c
	gcc -Wall -Wextra -Wno-unused-parameter -c $^ -o $@

%: %.o
	gcc $^ -o $@

.PHONY: clean

clean:
	rm -f cdecl computer_dating *.o
