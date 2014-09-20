#ifndef MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_

#include <vector>

/*
 * A divide and conquer template, in the style of an algorithmic
 * skeleton. The template defines a number of functions which are used
 * to implement divide and conquer behaviour. Individual template
 * specialisations can flesh out these "muscle" functions in order to
 * provide application specific logic:
 *
 *      bool isIndivisible(T) - Determine whether "T" can be solved
 *      T    solve(T)         - Solve "T", where "T" is indivisible
 *      T    merge(T[])       - Merge multiple "T"s together
 *      T[]  split(T)         - Split single "T" into 2 or more "T"s
 */
template<class T>
class DaC {

 public:
    // Constructor. If "lazy_eval" is true, don't process data until
    // DaC::get() is invoked.
    DaC(T data, bool lazy_eval = false);

    // Return the processed data. Will block until data is ready.
    T get();

    /*
     * Muscle functions:
     */

    // Determine whether "T" can be solved
    bool isIndivisible(T);

    // Split "T" into 2 or more "T"s
    std::vector<T> split(T);

    // Solve "T", where "T" is indivisible
    T solve(T data);

    // Merge multiple "T"s together
    T merge(std::vector<T>);


 protected:
    // Internal data:
    enum {
        IDLE,
        PROCESSING,
        READY
    } data_status;
    T data;
    T _dac(T data);
    void _run();
};


/*
 * Constructor definition.
 */
template<class T>
DaC<T>::DaC(T data, bool lazy_eval) {
    this->data = data;

    if (lazy_eval)
        this->data_status = IDLE;
    else
        _run();
};

/*
 * Return the processed data. Will block until data is ready.
 */
template<class T>
T DaC<T>::get() {
    if (this->data_status == IDLE)
        _run();

    while (this->data_status != READY)
        ;
    return this->data;
}


/*
 * The divide and conquer algorithm definition.
 */
template<class T>
T DaC<T>::_dac(T data) {
    if (isIndivisible(data))
        return solve(data);
    else {
        std::vector<T> split_data = split(data);

        for (std::vector<int>::size_type i = 0; i < split_data.size(); i++)
            split_data[i] = _dac(split_data[i]);

        return merge(split_data);
    }

    return data;
};


/*
 * Run the divide and conquer algorithm.
 */
template<class T>
void DaC<T>::_run() {
    this->data_status = PROCESSING;
    this->data = _dac(this->data);
    this->data_status = READY;
}


/*
 * Default solution implementation. Simply return data.
 */
template<class T>
T DaC<T>::solve(T data) {
    return data;
}


#endif // MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_
