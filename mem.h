#ifndef MEM_H
#define MEM_H
#include <stddef.h>

void* my_malloc(size_t size);

void my_free(void* ptr);

void print_memory_usage();

#endif //MEM_H
