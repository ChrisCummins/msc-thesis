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
    vector_t *const left = &in[0];
    vector_t *const right = &in[1];
    const typename vector_t::size_t length = left->length + right->length;

    // Allocate space for merged data:
    out->data = new T[length];
    out->length = length;

    unsigned int l = 0, r = 0, i = 0;

    /*
     * Iterate through the "left" and "right" input vectors, sorting
     * them. Where two elements are of the same value, the "left"
     * element will be placed first (stable sort).
     */
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
}

#endif // MSC_THESIS_EXERCISES_TEMPLATES_MERGE_SORT_H_
