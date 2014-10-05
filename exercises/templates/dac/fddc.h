#ifndef EXERCISES_TEMPLATES_DAC_FDDC_H_
#define EXERCISES_TEMPLATES_DAC_FDDC_H_

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>  // NOLINT(readability/streams)
#include <thread>
#include <vector>

#include "./debug.h"
#include "./vector.h"

/*
 * Abstract fixed degree Divide and Conquer skeleton.
 *
 * Provides a template for writing divide and conquer solutions by
 * overriding the 4 "muscle" functions. Solution classes must inherit
 * from this base class and (at a minimum) specify a merge function
 * (there are default implementations of solve, split, and
 * isIndivisible).
 *
 * divide_and_conquer() acts as the "worker" function, invoking the
 * muscle functions as required. When invoked, it determines whether
 * the size of the input is small enough to "solve" directly, and if
 * not, it splits the problem and invokes itself recursively "k"
 * times, where "k" is the number of sub-problems created by a split
 * operation (aka. the degree).
 *
 * Primitive concurrency is provided by measuring the depth of
 * recursion in the divide_and_conquer() function. If the depth is
 * less than the configurable "parallelisation_depth" parameter, then
 * recursion will occur in a new thread. This means that the maximum
 * total number of threads created is determined by both the degree
 * and the parallelisation depth. For a fixed degree of 2:
 *
 *             parallelisation_depth   max_no_of_threads
 *                                 0   1
 *                                 1   3
 *                                 2   7
 *                                 3   15
 *                                 4   31
 *
 * Or, more generally: n^k + (n-1)^k + (n-2)^k + ... + 1
 *
 */

template<class T>
class FDDC {
 public:
    typedef vector<T> vector_t;

    FDDC(vector_t *const data_in, const unsigned int degree);

    // Configurable parameters:

    // Maximum depth at which each recursion spawns a new
    // thread. Higher values means more concurrent execution, lower
    // values means more sequential. A value of 1 means no currency.
    void set_parallelisation_depth(const unsigned int n);


    void run();
    vector_t *get();


    // "Muscle" functions:
    virtual bool isIndivisible(const vector_t &t);                    // T -> b
    virtual void solve(vector_t *const in);                           // T -> T
    virtual void split(vector_t *const in, vector_t *const out);      // T-> T[]
    virtual void merge(vector_t *const in, vector_t *const out) = 0;  // T[]-> T


 protected:
    void divide_and_conquer(vector_t *const in, const unsigned int depth = 0);
    vector_t *const data;
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
void FDDC<T>::solve(vector_t *const in) {}


template<class T>
void FDDC<T>::split(vector_t *const in, vector_t *const out) {
    const typename vector_t::size_t split_length
        = in->length / this->split_degree;
    const typename vector_t::size_t first_split_length
        = in->length - (this->split_degree - 1) * split_length;

    // Split "in" into "k" vectors, starting at address "out".
    out[0].data = in->data;
    out[0].length = first_split_length;

    for (unsigned int i = 1; i < this->split_degree; i++) {
        const unsigned int offset = (i-1) * split_length + first_split_length;

        out[i].data = &in->data[offset];
        out[i].length = split_length;
    }
}


template<class T>
void FDDC<T>::divide_and_conquer(vector_t *const in,
                                 const unsigned int depth) {
    if (isIndivisible(*in)) {
        solve(in);
    } else {
        const int next_depth = depth + 1;
        const unsigned int k = this->split_degree;

        /*
         * Allocate a contiguous block of vectors for processing. This
         * size of the block is 2*k, and is divided into pre/post
         * recursion pairs such that each pair has the indexes:
         *
         *     buf[n], buf[n+k].
         */
        vector_t *const buf = new vector_t[k];

        // Split "in" to vectors buf[0:k]
        split(in, buf);

        /*
         * Recurse and solve for all sub-problems created by split().
         *
         * If the depth is less than "parallelisation_depth", then we
         * create a new thread to perform the recursion in. Otherwise,
         * we recurse sequentially.
         */
        if (depth < this->parallelisation_depth) {
            std::vector<std::thread> threads(k);

            // Create threads:
            for (unsigned int i = 0; i < k; i++) {
                threads[i] = std::thread(&FDDC<T>::divide_and_conquer, this,
                                         &buf[i], next_depth);
                // Debugging analytics
                IF_DAC_DEBUG(this->thread_count++);
                IF_DAC_DEBUG(this->active_thread_count++);
                DAC_DEBUG_PRINT(3, "Creating thread " << this->thread_count
                                << " at depth " << depth
                                << " (" << this->active_thread_count
                                << " active)");
            }

            // Block until threads complete:
            for (auto &thread : threads) {
                thread.join();
                // Debugging analytics
                IF_DAC_DEBUG(this->active_thread_count--);
                DAC_DEBUG_PRINT(3, "Thread completed at depth " << depth <<
                                " (" << this->active_thread_count
                                << " still active)");
            }

        } else {
            // Sequential:
            for (unsigned int i = 0; i < k; i++) {
                divide_and_conquer(&buf[i], next_depth);
            }
        }

        // Merge buffers buf[k:2k] into "out":
        merge(buf, in);

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
    return this->data;
}


template<class T>
void FDDC<T>::run() {
    DAC_ASSERT(this->active_thread_count == 1);

    divide_and_conquer(this->data);

    DAC_ASSERT(this->active_thread_count == 1);
    DAC_DEBUG_PRINT(2, "Number of threads created: " << this->thread_count);
}


// Constructor
template<class T>
FDDC<T>::FDDC(vector_t *const in, const unsigned int degree)
: data(in), split_degree(degree) {
    parallelisation_depth = 0;

    IF_DAC_DEBUG(this->thread_count = 1);
    IF_DAC_DEBUG(this->active_thread_count = 1);
}


#endif  // EXERCISES_TEMPLATES_DAC_FDDC_H_
