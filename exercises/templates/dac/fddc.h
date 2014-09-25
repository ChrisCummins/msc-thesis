#ifndef MSC_THESIS_EXERCISES_TEMPLATES_FDDC_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_FDDC_H_

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "debug.h"
#include "vector.h"


/*****************************************************/
/* Abstract Fixed degree Divide and Conquer skeleton */
/*****************************************************/

template<class T>
class FDDC {
 public:
    typedef vector<T> vector_t;

    // Constructor
    FDDC(vector_t *const data_in, const unsigned int degree);

    // Configurable parameters:

    // Maximum depth at which each recursion spawns a new
    // thread. Higher values means more concurrent execution, lower
    // values means more sequential. A value of 1 means no currency.
    void set_parallelisation_depth(const unsigned int n);

    void run();
    vector_t *get();


    // "Muscle" functions:
    virtual bool isIndivisible(const vector_t &t);                   // T   -> bool
    virtual void solve(vector_t *const in, vector_t *const out);     // T   -> T
    virtual void split(vector_t *const in, vector_t *const out);     // T   -> T[]
    virtual void merge(vector_t *const in, vector_t *const out) = 0; // T[] -> T


 protected:
    void divide_and_conquer(vector_t *const in, vector_t *const out,
                            const unsigned int depth = 0);
    vector_t *const data_in;
    vector_t *const data_out;
    unsigned int split_degree;
    unsigned int parallelisation_depth;

#ifdef DAC_DEBUG
    std::atomic_uint thread_count;
    std::atomic_uint active_thread_count;
#endif
};


/*******************************************/
/* Divide and conquer skeleton definitions */
/*******************************************/

template<class T>
bool FDDC<T>::isIndivisible(const vector_t &d) {
    return d.length <= 1;
}


template<class T>
void FDDC<T>::solve(vector_t *const in, vector_t *const out) {
    // Copy vector contents:
    out->copy(in);
}


template<class T>
void FDDC<T>::split(vector_t *const in, vector_t *const out) {
    const typename vector_t::size_t split_size = in->length / this->split_degree;

    // Split "in" into "k" vectors, starting at address "out".
    for (unsigned int i = 0; i < this->split_degree; i++) {
        const unsigned int offset = i * split_size;
        typename vector_t::size_t length = split_size;

        // Add on remainder if not an even split:
        if (i == this->split_degree - 1 && in->length % split_size)
            length += in->length % split_size;

        // Copy memory from one vector to another:
        out[i].copy(in, offset, length);
    }
}


template<class T>
void FDDC<T>::divide_and_conquer(vector_t *const in, vector_t *const out,
                               const unsigned int depth) {

    if (isIndivisible(*in)) {

        solve(in, out);

    } else {
        const int next_depth = depth + 1;
        const unsigned int k = this->split_degree;

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
        if (depth < this->parallelisation_depth) {
            std::thread threads[k];

            // Create threads:
            for (unsigned int i = 0; i < k; i++) {
                threads[i] = std::thread(&FDDC<T>::divide_and_conquer, this,
                                         &buf[i], &buf[i+k], next_depth);
                // Debugging analytics
                IF_DAC_DEBUG(this->thread_count++);
                IF_DAC_DEBUG(this->active_thread_count++);
                DAC_DEBUG_PRINT(3, "Creating thread " << this->thread_count
                                << " at depth " << depth
                                << " (" << this->active_thread_count << " active)");
            }

            // Block until threads complete:
            for (auto &thread : threads) {
                thread.join();
                // Debugging analytics
                IF_DAC_DEBUG(this->active_thread_count--);
                DAC_DEBUG_PRINT(3, "Thread completed at depth " << depth <<
                                " (" << this->active_thread_count << " still active)");
            }

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
void FDDC<T>::set_parallelisation_depth(const unsigned int n) {
    this->parallelisation_depth = n;
}

template<class T>
vector<T> *FDDC<T>::get() {
    return this->data_out;
}


template<class T>
void FDDC<T>::run() {
    DAC_ASSERT(this->active_thread_count == 1);

    divide_and_conquer(this->data_in, this->data_out);

    DAC_ASSERT(this->active_thread_count == 1);
    DAC_DEBUG_PRINT(2, "Number of threads created: " << this->thread_count);
}


// Constructor
template<class T>
FDDC<T>::FDDC(vector_t *const in, const unsigned int degree)
: data_in(in), data_out(new vector_t), split_degree(degree) {
    parallelisation_depth = 0;

    IF_DAC_DEBUG(this->thread_count = 1);
    IF_DAC_DEBUG(this->active_thread_count = 1);
}


#endif // MSC_THESIS_EXERCISES_TEMPLATES_FDDC_H_
