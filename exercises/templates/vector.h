#ifndef MSC_THESIS_EXERCISES_TEMPLATES_VECTOR_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_VECTOR_H_

/**************************/
/* "Vector" storage class */
/**************************/

template<class T>
class _vector {
 public:
    typedef unsigned int size_t;

    _vector() {};
    _vector(const _vector<T>::size_t length);
    _vector(T *const data, const _vector<T>::size_t length);
    ~_vector();

    T     *data;
    _vector<T>::size_t length;

    void print();
    bool isSorted(bool quiet = false);
    void copy(_vector<T> *const src);
    void copy(_vector<T> *const src, size_t n);
    void copy(_vector<T> *const src, size_t offset, size_t n);
};


/************************/
/* Template definitions */
/************************/

template<class T>
_vector<T>::_vector(const _vector<T>::size_t length) {
    this->data = new T[length];
    this->length = length;
};


template<class T>
_vector<T>::_vector(T *const data, const _vector<T>::size_t length) {
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

    for (_vector<T>::size_t i = 1; i < this->length; i++) {
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


/*************************/
/* Vector copy functions */
/*************************/

template<class T>
inline void _vector<T>::copy(_vector<T> *const src) {
    this->copy(src, 0, src->length);
}

template<class T>
inline void _vector<T>::copy(_vector<T> *const src, const size_t n) {
    this->copy(src, 0, n);
}

template<class T>
inline void _vector<T>::copy(_vector<T> *const src, const size_t offset, const size_t n) {
    this->data = new T[n];
    memcpy(this->data, src->data + offset, n * sizeof(*src->data));
    this->length = n;
}

#endif // MSC_THESIS_EXERCISES_TEMPLATES_VECTOR_H_
