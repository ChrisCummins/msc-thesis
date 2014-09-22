#include <cstdlib>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>


template<class T>
class vector {
public:
    T *data;
    unsigned int length;

    ~vector() {
        delete[] this->data;
    }
};

typedef vector<int> data;

void print_vector(const data &d) {
    std::printf("%14p length: %2d, data: { ", d.data, d.length);

    const unsigned int max = d.length < 10 ? d.length : 10;

    for (unsigned int i = 0; i < max; i++)
        std::cout << d.data[i] << " ";

    if (d.length > max)
        std::cout << "...";
    else
        std::cout << "}";

    std::cout << "\n";
}

bool isIndivisible(data *const d) {
    return d->length <= 1;
};


void split(data *const in, data *const left, data *const right) {
    const int right_len  = in->length / 2;
    const int left_len = in->length - right_len;

    left->data = new int[left_len];
    memcpy(left->data, in->data, left_len * sizeof(*in->data));
    left->length = left_len;

    right->data = new int[right_len];
    memcpy(right->data, in->data + left_len, right_len * sizeof(*in->data));
    right->length = right_len;
};


void solve(data *const in, data *const out) {
    out->data = new int[in->length];
    memcpy(out->data, in->data, in->length * sizeof(*in->data));
    out->length = in->length;
};


void merge(const data &left, const data &right, data *const out) {
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
};

void divide_and_conquer(data *const in, data *const out, const int depth = 0) {

    if (isIndivisible(in)) {
        solve(in, out);
    } else {
        const int next_depth = depth + 1;

        // Allocate data structures on heap:
        data *const buf       = new data[4];
        data *const in_left   = &buf[0];
        data *const in_right  = &buf[1];
        data *const out_left  = &buf[2];
        data *const out_right = &buf[3];

        // Split, recurse, and merge:
        split(in, in_left, in_right);
        divide_and_conquer(in_left,  out_left,  next_depth);
        divide_and_conquer(in_right, out_right, next_depth);
        merge(*out_left, *out_right, out);
        
        delete[] buf;
    }
}

#define TEST_SIZE 2100000

int main(int argc, char *argv[]) {
    int nums[TEST_SIZE];

    for (int i = 0; i < TEST_SIZE; i++)
        nums[i] = rand() % TEST_SIZE;

    // TODO: compare against:
    //std::sort(nums, nums+TEST_SIZE);

    data *const a = new data;
    data *const b = new data;

    a->data   = &nums[0];
    a->length = sizeof(nums) / sizeof(nums[0]);

    print_vector(*a);
    divide_and_conquer(a, b);
    print_vector(*b);

    delete b;

    return 0;
}
