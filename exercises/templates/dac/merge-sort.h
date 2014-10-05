#ifndef EXERCISES_TEMPLATES_DAC_MERGE_SORT_H_
#define EXERCISES_TEMPLATES_DAC_MERGE_SORT_H_

#include <algorithm>

#include "./fddc.h"

/*
 * A concrete stable merge sort implementation, using the Divide and
 * Conquer skeleton.
 *
 * The only muscle function we need to provide is merge(), since the
 * defaults for the FDDC template are satisfactory.
 *
 * MergeSort requires that the class "T" supports comparion and
 * equality operators.
 */

template<class T>
class MergeSort : public FDDC<T> {
    using vector_t = typename FDDC<T>::vector_t;

 public:
    explicit MergeSort(vector_t *const data_in);
    bool isIndivisible(const vector_t &t);
    void solve(vector_t *const in);
    void merge(vector_t *const in, vector_t *const out);

    // The threshold at which vectors are sorted either by splitting
    // and recursion, or by insertion sort.
    void set_split_threshold(typename vector_t::size_t split_threshold);

 protected:
    typename vector_t::size_t split_threshold;
};

// Constructor. Call base skeleton with fixed degree
template<class T>
MergeSort<T>::MergeSort(vector_t *const data_in)
: FDDC<T>(data_in, 2) {}


template<class T>
bool MergeSort<T>::isIndivisible(const vector_t &in) {
    return in.length <= this->split_threshold;
}


// Perform insertion sort on vector "in"
template<class T>
void MergeSort<T>::solve(vector_t *const in) {
    T key;
    typename vector_t::size_t j;

    for (typename vector_t::size_t i = 1; i < in->length; i++) {
        key = in->data[i];
        j = i;

        while (j > 0 && in->data[j - 1] > key) {
            in->data[j] = in->data[j - 1];
            j--;
        }
        in->data[j] = key;
    }
}


template<class T>
void MergeSort<T>::merge(vector_t *const in, vector_t *const out) {
    const typename vector_t::size_t n1 = in[0].length;
    const typename vector_t::size_t n2 = in[1].length;

    // Make local copies of input vectors on stack:
    T L[n1];
    T R[n2];

    std::copy(in[0].data, in[0].data + n1, &L[0]);
    std::copy(in[1].data, in[1].data + n2, &R[0]);

    typename vector_t::size_t i = 0, l = 0, r = 0;

    // Setup output vector:
    out->data = in[0].data;
    out->length = n1 + n2;

    /*
     * Iterate through the "left" and "right" input vectors, sorting
     * them. Where two elements are of the same value, the "left"
     * element will be placed first (stable sort).
     */
    while (l < n1 && r < n2) {
        if (R[r] < L[l])
            out->data[i++] = R[r++];
        else
            out->data[i++] = L[l++];
    }

    while (r < n2)
        out->data[i++] = R[r++];

    while (l < n1)
        out->data[i++] = L[l++];
}


template<class T>
void MergeSort<T>::set_split_threshold(
    typename vector_t::size_t split_threshold) {
  this->split_threshold = split_threshold;
}

#endif  // EXERCISES_TEMPLATES_DAC_MERGE_SORT_H_
