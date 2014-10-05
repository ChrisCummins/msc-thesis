#include "test.h"

namespace {

    int *get_rand_int_array(const size_t size) {
        int *const a = new int[size];

        for (size_t i = 0; i < size; i++)
            a[i] = rand() % size;

        return a;
    }

    float *get_rand_float_array(const size_t size) {
        float *const a = new float[size];

        for (size_t i = 0; i < size; i++)
            a[i] = ((float)rand() / (float)size);

        return a;
    }

    template<class T>
    vector<T> *get_unsorted_vector(const size_t size, T *(*get_array)(const size_t)) {
        T *const a = get_array(size);
        vector<T> *const v = new vector<T>;

        v->data = a;
        v->length = size;

        return v;
    }
}

int *get_unsorted_int_array(const size_t size) {
    return get_rand_int_array(size);
}

vector<int> *get_unsorted_int_vector(const size_t size) {
    return get_unsorted_vector(size, get_rand_int_array);
}

vector<float> *get_unsorted_float_vector(const size_t size) {
    return get_unsorted_vector(size, get_rand_float_array);
}
