#ifndef MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "timer.h"
#include "vector.h"

// Maximum depth at which each recursion spawns a new thread. Higher
// values means more concurrent execution, lower values means more
// sequential. A value of 1 means no currency:
#define FORK_DEPTH 4


/*******************************************/
/* Fixed depth Divide and Conquer skeleton */
/*******************************************/

template<class T>
class DC {
 public:
    typedef vector<T> vector_t;

    DC(vector_t *const data);

    vector_t *get();

    // "Muscle" functions:
    bool isIndivisible(const vector_t &);                 // T   -> bool
    void solve(vector_t *const in, vector_t *const out);  // T   -> T
    void split(vector_t *const in, vector_t *const out);  // T   -> T[]
    void merge(vector_t *const in, vector_t *const out);  // T[] -> T

 private:
    void divide_and_conquer(vector_t *const in, vector_t *const out,
                            const int depth = 0);
    vector_t *data;
    unsigned int k;
};


/*******************************************/
/* Divide and conquer skeleton definitions */
/*******************************************/

template<class T>
bool DC<T>::isIndivisible(const vector_t &d) {
    return d.length <= 1;
}


template<class T>
void DC<T>::solve(vector_t *const in, vector_t *const out) {
    // Copy vector contents:
    out->copy(in);
}


template<class T>
void DC<T>::split(vector_t *const in, vector_t *const out) {
    const typename vector_t::size_t split_size = in->length / this->k;

    // Split "in" into "k" vectors, starting at address "out".
    for (unsigned int i = 0; i < this->k; i++) {
        const unsigned int offset = i * split_size;
        typename vector_t::size_t length = split_size;

        // Add on remainder if not an even split:
        if (i == this->k - 1 && in->length % split_size)
            length += in->length % split_size;

        // Copy memory from one vector to another:
        out[i].copy(in, offset, length);
    }
}

template<class T>
void DC<T>::divide_and_conquer(vector_t *const in, vector_t *const out,
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
        vector_t *const buf = new vector_t[k * 2];


        // Split, recurse, and merge:
        split(in, buf);

        /*
         * If the depth is less than some arbitrary value, then we
         * create a new thread to perform the recursion in. Otherwise,
         * we recurse sequentially.
         */
        if (depth < FORK_DEPTH) {

            std::thread threads[k];

            // Create threads:
            for (unsigned int i = 0; i < k; i++)
                threads[i] = std::thread(&DC<T>::divide_and_conquer, this,
                                         &buf[i], &buf[i+k], next_depth);

            // Block until threads complete:
            for (auto &thread : threads)
                thread.join();

        } else {

            // Sequential:
            for (unsigned int i = 0; i < k; i++)
                divide_and_conquer(&buf[i], &buf[i+k], next_depth);

        }

        merge(&buf[k], out);

        // Free heap memory:
        delete[] buf;
    }
}


template<class T>
vector<T> *DC<T>::get() {
    return this->data;
}


template<class T>
DC<T>::DC(vector_t *const in) {
    this->data = new vector_t;
    // TODO: Assign as a constructor parameter
    this->k = 2;
    divide_and_conquer(in, this->data);
}


/*****************************/
/* MergeSort specialisations */
/*****************************/

template<class T>
void DC<T>::merge(vector_t *const in, vector_t *const out) {
    vector_t *const left = &in[0];
    vector_t *const right = &in[1];
    const typename vector_t::size_t length = left->length + right->length;

    out->data = new T[length];

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

#endif // MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_
