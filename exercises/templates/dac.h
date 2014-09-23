#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>

#include "timer.h"

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
};


template<class T>
class DC {
 public:
    typedef _vector<T> vector;

    DC(vector *const);

    vector *get();
    bool isIndivisible(const vector &);
    void split(vector *const in, vector *const left, vector *const right);
    void solve(vector *const in, vector *const out);
    void merge(const vector &left, const vector &right, vector *const out);

 private:
    void divide_and_conquer(vector *const in, vector *const out, const int depth = 0);
    vector *data;
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
};

template<class T>
bool DC<T>::isIndivisible(const vector &d) {
    return d.length <= 1;
}

template<class T>
void DC<T>::split(vector *const in, vector *const left, vector *const right) {
    const int right_len  = in->length / 2;
    const int left_len = in->length - right_len;

    left->data = new int[left_len];
    memcpy(left->data, in->data, left_len * sizeof(*in->data));
    left->length = left_len;

    right->data = new int[right_len];
    memcpy(right->data, in->data + left_len, right_len * sizeof(*in->data));
    right->length = right_len;
}

template<class T>
void DC<T>::solve(vector *const in, vector *const out) {
    out->data = new int[in->length];
    memcpy(out->data, in->data, in->length * sizeof(*in->data));
    out->length = in->length;
}

template<class T>
void DC<T>::merge(const vector &left, const vector &right, vector *const out) {
    const int length = left.length + right.length;

    out->data = new int[length];

    unsigned int l = 0, r = 0, i = 0;

    while (l < left.length && r < right.length) {
        if (right.data[r] < left.data[l])
            out->data[i++] = right.data[r++];
        else
            out->data[i++] = left.data[l++];
    }

    while (r < right.length)
        out->data[i++] = right.data[r++];

    while (l < left.length)
        out->data[i++] = left.data[l++];

    out->length = length;
}

template<class T>
void DC<T>::divide_and_conquer(vector *const in, vector *const out, const int depth) {

    if (isIndivisible(*in)) {
        solve(in, out);
    } else {
        const int next_depth = depth + 1;

        // Allocate data structures on heap:
        vector *const buf       = new vector[4];
        vector *const in_left   = &buf[0];
        vector *const in_right  = &buf[1];
        vector *const out_left  = &buf[2];
        vector *const out_right = &buf[3];

        // Split, recurse, and merge:
        split(in, in_left, in_right);
        divide_and_conquer(in_left,  out_left,  next_depth);
        divide_and_conquer(in_right, out_right, next_depth);
        merge(*out_left, *out_right, out);

        delete[] buf;
    }
}

template<class T>
DC<T>::DC(vector *const in) {
    this->data = new vector;
    divide_and_conquer(in, this->data);
}
