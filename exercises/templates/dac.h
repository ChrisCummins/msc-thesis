#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "timer.h"

#define FORK_DEPTH 4

template<class T>
class _vector {
 public:
    typedef unsigned int size_t;

    _vector() {};
    _vector(const size_t length);
    _vector(T *const data, const size_t length);
    ~_vector();

    T     *data;
    size_t length;

    void print();
    bool isSorted(bool quiet = false);
};


template<class T>
class DC {
 public:
    typedef _vector<T> vector;

    DC(vector *const);

    vector *get();
    bool isIndivisible(const vector &);
    void split(vector *const in, vector *const out);
    void solve(vector *const in, vector *const out);
    void merge(vector *const in, vector *const out);

 private:
    void divide_and_conquer(vector *const in, vector *const out, const int depth = 0);
    vector *data;
    unsigned int k;
};


/************************/
/* Template definitions */
/************************/

template<class T>
_vector<T>::_vector(const size_t length) {
    this->data = new T[length];
    this->length = length;
};

template<class T>
_vector<T>::_vector(T *const data, const size_t length) {
    this->data = data;
    this->length = length;
}

template<class T>
_vector<T>::~_vector() {
    delete[] this->data;
}

template<class T>
void _vector<T>::print() {
    std::printf("%14p length: %2d, data: { ", this->data, this->length);

    const unsigned int max = this->length < 10 ? this->length : 10;

    for (unsigned int i = 0; i < max; i++)
        std::cout << this->data[i] << " ";

    if (this->length > max)
        std::cout << "...";
    else
        std::cout << "}";

    std::cout << "\n";
}

template<class T>
bool _vector<T>::isSorted(bool quiet) {

    for (size_t i = 1; i < this->length; i++) {
        if (this->data[i] < this->data[i - 1]) {
            if (!quiet) {
                this->print();
                std::cout << "List item " << i << " is not sorted\n";
            }
            return false;
        }
    }

    return true;
}

template<class T>
bool DC<T>::isIndivisible(const vector &d) {
    return d.length <= 1;
}

template<class T>
void DC<T>::split(vector *const in, vector *const out) {
    const typename vector::size_t split_size = in->length / this->k;

    // Split "in" into "k" vectors, starting at address "out".
    for (unsigned int i = 0; i < this->k; i++) {
        const unsigned int offset = i * split_size;
        typename vector::size_t length = split_size;

        // Add on remainder if not an even split:
        if (i == this->k - 1 && in->length % split_size)
            length += in->length % split_size;

        const typename vector::size_t size = length * sizeof(*in->data);

        // Copy memory from one vector to another:
        out[i].data = new T[length];
        memcpy(out[i].data, in->data + offset, size);
        out[i].length = length;
    }
}

template<class T>
void DC<T>::solve(vector *const in, vector *const out) {
    out->data = new int[in->length];
    memcpy(out->data, in->data, in->length * sizeof(*in->data));
    out->length = in->length;
}

template<class T>
void DC<T>::merge(vector *const in, vector *const out) {
    vector *const left = &in[0];
    vector *const right = &in[1];
    const typename vector::size_t length = left->length + right->length;

    out->data = new int[length];

    unsigned int l = 0, r = 0, i = 0;

    while (l < left->length && r < right->length) {
        if (right->data[r] < left->data[l])
            out->data[i++] = right->data[r++];
        else
            out->data[i++] = left->data[l++];
    }

    while (r < right->length)
        out->data[i++] = right->data[r++];

    while (l < left->length)
        out->data[i++] = left->data[l++];

    out->length = length;
}

template<class T>
void DC<T>::divide_and_conquer(vector *const in, vector *const out,
                               const int depth) {

    if (isIndivisible(*in)) {

        solve(in, out);

    } else {
        const int next_depth = depth + 1;
        const unsigned int k = this->k;

        /*
         * Allocate a contiguous block of vectors for processing. This
         * size of the block is 2*k, and is divided into pre/post
         * recursion pairs such that each pair has the indexes buf[n],
         * buf[n+k].
         */
        vector *const buf = new vector[k * 2];

        // Split, recurse, and merge:
        split(in, buf);

        /*
         * If the depth is less than some arbitrary value, then we
         * create a new thread to perform the recursion in. Otherwise,
         * we recurse sequentially.
         */
        if (depth < FORK_DEPTH) {

            // Multi-threaded:
            std::thread threads[k];

            for (unsigned int i = 0; i < k; i++)
                threads[i] = std::thread(&DC<T>::divide_and_conquer, this,
                                         &buf[i], &buf[i+k], next_depth);

            for (unsigned int i = 0; i < k; i++)
                threads[i].join();

        } else {

            // Sequential:
            for (unsigned int i = 0; i < k; i++)
                divide_and_conquer(&buf[i], &buf[i+k], next_depth);

        }

        merge(&buf[k], out);

        /*
         * Free vector buffer:
         */
        delete[] buf;
    }
}

template<class T>
_vector<T> *DC<T>::get() {
    return this->data;
}

template<class T>
DC<T>::DC(vector *const in) {
    this->data = new vector;
    // TODO: Assign as a constructor parameter
    this->k = 2;
    divide_and_conquer(in, this->data);
}
