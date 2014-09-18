#include "stack.h"

// Test:
int main(int argc, char *argv[]) {

    Stack<int> s;

    std::cout << s.toString() << "\n";

    for (int i = 1; i <= 10; i++)
        s.push(i);

    std::cout << s.toString() << "\n";

    std::cout << "               peek() = " << s.peek() << "\n";
    std::cout << s.toString() << "\n";

    do {
        std::cout << "               pop() = " << s.pop() << "\n";
        std::cout << s.toString() << "\n";
    } while (!s.empty());

    return 0;
}
