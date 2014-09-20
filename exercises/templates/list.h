#ifndef MSC_THESIS_EXERCISES_TEMPLATES_LIST_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_LIST_H_

#include <vector>
#include <iostream>

#define DEFAULT_LIST_SIZE 100000

typedef std::vector<int>  List;
typedef std::vector<List> Lists;


List get_rand_list(size_t size = DEFAULT_LIST_SIZE);

void print_list(List list, bool truncate = true);
bool list_is_sorted(List list);


#endif // MSC_THESIS_EXERCISES_TEMPLATES_LIST_H_
