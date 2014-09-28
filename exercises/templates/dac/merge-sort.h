#ifndef MSC_THESIS_EXERCISES_TEMPLATES_MERGE_SORT_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_MERGE_SORT_H_

#include "fddc.h"

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

    MergeSort(vector_t *const data_in);
    void merge(vector_t *const in, vector_t *const out);
};

// Constructor. Call base skeleton with fixed degree
template<class T>
MergeSort<T>::MergeSort(vector_t *const data_in)
: FDDC<T>(data_in, 2) {}


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


#endif // MSC_THESIS_EXERCISES_TEMPLATES_MERGE_SORT_H_
