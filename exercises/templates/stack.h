#ifndef MSC_THESIS_EXERCISES_TEMPLATES_STACK_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_STACK_H_

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <stdexcept>
#include <sstream>

/*
 * A generic Stack type, implemented using a vector.
 *
 * Oh yeah, and ridiculous caching of values, cos why not.
 * Note: no actual performance gains under most circumstances.
 */
template<class T>
class Stack {

 public:

    Stack() {
        this->cache.size_dirty = true;
        this->cache.head_dirty = true;
    }

    void push(const T elem) {
        this->cache.size_dirty = true;
        this->cache.head = elem;

        stack.push_back(elem);
    };

    const T pop() {
        T back = this->peek();

        stack.pop_back();

        this->cache.size_dirty = true;
        this->cache.head_dirty = true;

        return back;
    };

    const T peek() {
        if (this->empty())
            throw std::out_of_range("Stack<>::peek() Empty stack");

        if (this->cache.head_dirty) {
            this->cache.head = stack.back();
            this->cache.head_dirty = false;
        }

        return this->cache.head;
    };

    const int length() {
        if (this->cache.size_dirty) {
            this->cache.size = stack.size();
            this->cache.empty = stack.empty();
            this->cache.size_dirty = false;
        }

        return this->cache.size;
    };

    const bool empty() {
        if (this->cache.size_dirty) {
            this->cache.size = stack.size();
            this->cache.empty = stack.empty();
            this->cache.size_dirty = false;
        }

        return stack.empty();
    };

    std::string toString() {
        std::stringstream fmt;
        fmt << "Stack[" << this << "]";
        return fmt.str();
    };

 private:
    std::vector<T> stack;

    struct {
        bool size_dirty;
        int size;
        bool empty;

        bool head_dirty;
        T head;
    } cache;
};


/*
 * Stack<int> specialization
 */
template <>
std::string Stack<int>::toString() {
    std::stringstream fmt;

    fmt << this << " Stack<int>[" << this->length() << "] = { ";

    for(std::vector<int>::size_type i = this->length() - 1;
        i != (std::vector<int>::size_type) - 1; i--) {
        fmt << stack[i] << " ";
    }

    fmt << "}";

    return fmt.str();
}

#endif // MSC_THESIS_EXERCISES_TEMPLATES_STACK_H_
