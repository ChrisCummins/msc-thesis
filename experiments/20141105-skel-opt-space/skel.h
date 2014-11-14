#ifndef EXPERIMENTS_20141105_SKEL_OPT_SPACE_SKEL_H_  // NOLINT(legal/copyright)
#define EXPERIMENTS_20141105_SKEL_OPT_SPACE_SKEL_H_

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>  // NOLINT(readability/streams)
#include <memory>
#include <thread>
#include <vector>
#include <future>

#include "./config.h"
#include "./debug.h"
#include "./nbits.h"

namespace skel {

// There are two types of divide_and_conquer skeleton:
// divide_and_conquer, and divide_and_transform. They both operate in
// the same way, except that divide_and_transform is an in-place
// operation, wheras divide_and_conquer skeletons return a value, with
// separate types for input and return value.


// Divide and Conquer skeleton.
//
// Provides a template for writing divide and transform solutions by
// providing 4 "muscle" functions:
//
// bool is_indivisible(const T& problem)
//
//    Returns whether a problem can be split into multiple
//    sub-problems, or whether the problem should be conquered
//    directly.
//
// std::vector<T> divide(const T& problem);
//
//    Divides the input problem into a set of smaller sub-problems
//    (when it can't be conquered directly).
//
// Q conquer(const T& problem)
//
//    Solve an indivisible problem directly.
//
// Q combine(std::vector<Q> solutions)
//
//    Combines a set of conquered problems ("solutions") into a single
//    solution.
//
template<typename Type, typename ReturnType,
    bool is_indivisible(const Type& problem),
    std::vector<Type> divide(const Type& problem),
    ReturnType conquer(const Type& problem),
    ReturnType combine(std::vector<ReturnType> problem)>
ReturnType divide_and_conquer(const Type& problem, const int depth = 0);


// Divide and Transform skeleton.
//
// Provides a template for writing divide and transform solutions by
// providing 4 "muscle" functions:
//
// bool is_indivisible(const T& problem)
//
//    Returns whether a problem can be split into multiple
//    sub-problems, or whether the problem should be conquered
//    directly.
//
// std::vector<T> divide(const T& problem);
//
//    Divides the input problem into a set of smaller sub-problems
//    (when it can't be conquered directly).
//
// void conquer(const T& problem)
//
//    Solve an indivisible problem directly.
//
// void combine(std::vector<T> solutions, T *const out)
//
//    Combines a set of conquered problems ("solutions") into a single
//    solution ("out").
//
template<typename Type,
    bool is_indivisible(const Type& problem),
    std::vector<Type> divide(const Type& problem),
    void transform(const Type& problem),
    void combine(std::vector<Type> solutions, Type *const out)>
void divide_and_transform(Type *const in, const int depth = 0);

// Shorthand because we're lazy:
#define DAC_SKEL_TEMPLATE_PARAMETERS \
    Type, ReturnType, is_indivisible, divide, conquer, combine
#define DAT_SKEL_TEMPLATE_PARAMETERS \
    Type, is_indivisible, divide, transform, combine

// Primitive concurrency is provided by measuring the depth of
// recursion in the divide_and_conquer() function. If the depth is
// less than a pre-determined "parallelisation depth", then recursion
// will occur in a new thread. This means that the maximum total
// number of threads created is coarsely determined by the
// parallelisation depth and the number of sub-problems which is
// generated by the divide() function, known as the degree. For a
// degree of 2:
//
//             parallelisation_depth   max_no_of_threads
//                                 0   1
//                                 1   3
//                                 2   7
//                                 3   15
//                                 4   31
//
// Or, more generally:
//
//     N = k^d - 1
//
// Where "d" is the parallelisation depth, and "k" is the
// degree. Rearranging this with respect to "d":
//
//     d = log_k(N+1)
//
// The ideal size of "N" will vary largely from application to
// application, but a reasonable average tends to be 2 times the
// number of available processors. If we assume that "k" is 2, we can
// use the pre-processor to give us a reasonable maximum
// parallelisaiton depth by using bit shifts:
//
#define DAC_SKEL_PARALLELISATION_DEPTH 4

//
// Divide and Conquer skeleton implementation.
////////////////////////////////////////////////////////////////////
//

template<typename Type, typename ReturnType,
    bool is_indivisible(const Type& problem),
    std::vector<Type> divide(const Type& problem),
    ReturnType conquer(const Type& problem),
    ReturnType combine(std::vector<ReturnType> problem)>
ReturnType divide_and_conquer(const Type& problem, const int depth) {
// Cheeky shorthand:
#define self divide_and_conquer<DAC_SKEL_TEMPLATE_PARAMETERS>

  // Determine whether we're in a base case or recursion case:
  if (is_indivisible(problem)) {
    // If we're in a base case, then we can solve the problem
    // directly:
    return conquer(problem);
  } else {
    // If we're in a recursion case, then we need to divide the
    // problem into multiple subproblems, and recurse on each of those
    // sub-problems, before combining the results:
    const int next_depth = depth + 1;

    std::vector<Type> sub_problems = divide(problem);
    const int degree = sub_problems.size();
    std::vector<ReturnType> solved_problems(degree);

    // Debugging output:
#if DAC_DEBUG_LEVEL >= 2
    if (depth == 0) {
#  if DAC_SKEL_PARALLELISATION_DEPTH > 0
      DAC_DEBUG_PRINT(2, "Using parallelisation depth "
                      << DAC_SKEL_PARALLELISATION_DEPTH)
#  else
      DAC_DEBUG_PRINT(2, "Using sequential skeleton back-end")
#  endif  // DAC_SKEL_PARALLELISATION_DEPTH
    }
#endif  // DAC_DEBUG_LEVEL


// If the parallelisation depth is set greater than 0, then it means
// we may be recursing in parallel. Otherwise, we will *always*
// recurse sequentially. We can use the pre-processor to optimise for
// this case by only including the conditional logic in the when the
// condition is actually needed.
#if DAC_SKEL_PARALLELISATION_DEPTH > 0

    // If the current recursion depth is less than the parallelisation
    // depth, we create new threads to perform the recursion
    // in. Otherwise, recurse sequentially.
    if (depth < DAC_SKEL_PARALLELISATION_DEPTH) {
      std::vector<std::future<ReturnType>> threads(sub_problems.size());

      // Parallelised section. Create threads and block until
      // completed:
      for (int i = 0; i < degree; i++) {
        threads[i] = std::async(&self, sub_problems[i], next_depth);
        DAC_DEBUG_PRINT(3, "Creating thread at depth " << next_depth);
      }
      for (int i = 0; i < degree; i++) {
        solved_problems[i] = threads[i].get();
        DAC_DEBUG_PRINT(3, "Thread completed at depth " << next_depth);
      }

    } else {
#endif  // DAC_SKEL_PARALLELISATION_DEPTH > 0

      // Sequential execution (*yawn*):
      for (int i = 0; i < degree; i++)
        solved_problems[i] = self(sub_problems[i], next_depth);

#if DAC_SKEL_PARALLELISATION_DEPTH > 0
    }
#endif  // DAC_SKEL_PARALLELISATION_DEPTH > 0

    return combine(solved_problems);
  }

#undef self
}

//
// Divide and Transform skeleton implementation.
////////////////////////////////////////////////////////////////////
//

template<typename Type,
    bool is_indivisible(const Type& problem),
    std::vector<Type> divide(const Type& problem),
    void transform(const Type& problem),
    void combine(std::vector<Type> problem, Type *const out)>
void divide_and_transform(Type *const problem, const int depth) {
// Cheeky shorthand:
#define self divide_and_transform<DAT_SKEL_TEMPLATE_PARAMETERS>

  // Determine whether we're in a base case or recursion case:
  if (is_indivisible(*problem)) {
    // If we're in a base case, then we can solve the problem
    // directly:
    transform(*problem);
  } else {
    // If we're in a recursion case, then we need to divide the
    // problem into multiple subproblems, and recurse on each of those
    // sub-problems, before combining the results:
    const int next_depth = depth + 1;

    std::vector<Type> sub_problems = divide(*problem);

    // Debugging output:
#if DAC_DEBUG_LEVEL >= 2
    if (depth == 0) {
#  if DAC_SKEL_PARALLELISATION_DEPTH > 0
      DAC_DEBUG_PRINT(2, "Using parallelisation depth "
                      << DAC_SKEL_PARALLELISATION_DEPTH)
#  else
      DAC_DEBUG_PRINT(2, "Using sequential skeleton back-end")
#  endif  // DAC_SKEL_PARALLELISATION_DEPTH
    }
#endif  // DAC_DEBUG_LEVEL


// If the parallelisation depth is set greater than 0, then it means
// we may be recursing in parallel. Otherwise, we will *always*
// recurse sequentially. We can use the pre-processor to optimise for
// this case by only including the conditional logic in the when the
// condition is actually needed.
#if DAC_SKEL_PARALLELISATION_DEPTH > 0

    // If the current recursion depth is less than the parallelisation
    // depth, we create new threads to perform the recursion
    // in. Otherwise, recurse sequentially.
    if (depth < DAC_SKEL_PARALLELISATION_DEPTH) {
      std::vector<std::thread> threads(sub_problems.size());

      // Parallelised section. Create threads and block until
      // completed:
      for (size_t i = 0; i < sub_problems.size(); i++) {
        threads[i] = std::thread(self, &sub_problems[i], next_depth);
        DAC_DEBUG_PRINT(3, "Creating thread at depth " << next_depth);
      }
      for (auto& thread : threads) {
        thread.join();
        DAC_DEBUG_PRINT(3, "Thread completed at depth " << next_depth);
      }

    } else {
#endif  // DAC_SKEL_PARALLELISATION_DEPTH > 0

      // Sequential execution (*yawn*):
      for (auto& sub_problem : sub_problems)
        self(&sub_problem, next_depth);

#if DAC_SKEL_PARALLELISATION_DEPTH > 0
    }
#endif  // DAC_SKEL_PARALLELISATION_DEPTH > 0

    combine(sub_problems, problem);
  }

#undef self
}


#undef DAC_SKEL_TEMPLATE_PARAMETERS
#undef DAT_SKEL_TEMPLATE_PARAMETERS

}  // namespace skel

#endif  // EXPERIMENTS_20141105_SKEL_OPT_SPACE_SKEL_H_