#include "mem.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef DEBUG_MEM
typedef struct entry {
    void* ptr;
    size_t size;
    int index;
    entry* prev;
    entry* next;
} entry;

bool hm_initialized = false;
int hm_num_buckets = 13;
entry** hm_buckets = NULL;

void hm_put(void* ptr, size_t size) {
    if (!hm_initialized) {
        hm_initialized = true;
        hm_buckets = (entry**) malloc(hm_num_buckets * sizeof(entry*));
        for (int i = 0; i < hm_num_buckets; i++) {
            hm_buckets[i] = NULL;
        }
    }

    int index = (int) ((uintptr_t) ptr % hm_num_buckets);
    entry* e = (entry*) malloc(sizeof(entry));
    e->ptr = ptr;
    e->size = size;
    e->index = index;
    e->prev = NULL;
    e->next = NULL;

    // printf("Adding entry: %p, size: %zu, index: %d\n", ptr, size, index);

    if (hm_buckets[index] == NULL) {
        hm_buckets[index] = e;
    } else {
        entry* current = hm_buckets[index];
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = e;
        e->prev = current;
    }
}

entry* hm_get(void* ptr) {
    if (hm_buckets == NULL) {
        return NULL;
    }
    int index = (int) ((uintptr_t) ptr % hm_num_buckets);
    entry* current = hm_buckets[index];
    while (current != NULL) {
        if (current->ptr == ptr) {
            return current;
        }
        current = current->next;
    }
    return NULL;
}

void hm_remove(entry* e) {
    if (hm_buckets[e->index] == e) {
        hm_buckets[e->index] = e->next;
    } else {
      if (e->prev != NULL) {
        e->prev->next = e->next;
      }
      if (e->next != NULL) {
          e->next->prev = e->prev;
      }
    }
    free(e);
}

size_t peak_memory = 0;
long long peak_allocs = 0;
size_t allocated_memory = 0;
long long allocated_count = 0;
#endif

void* my_malloc(size_t size) {
    void* ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

#ifdef DEBUG_MEM
    printf("Allocated %zu bytes at %p\n", size, ptr);

    hm_put(ptr, size);

    allocated_memory += size;
    if (allocated_memory > peak_memory) {
        peak_memory = allocated_memory;
    }
    allocated_count++;
    if (allocated_count > peak_allocs) {
        peak_allocs = allocated_count;
    }
#endif

    return ptr;
}

void my_free(void* ptr) {
    if (ptr == NULL) {
        return;
    }
    free(ptr);

#ifdef DEBUG_MEM
    entry* e = hm_get(ptr);
    if (e == NULL) {
        fprintf(stderr, "Memory %p not found in hash map\n", ptr);
        return;
    }

    allocated_memory -= e->size;
    allocated_count--;
    if (allocated_count < 0) {
        fprintf(stderr, "Allocated count is negative\n");
        exit(EXIT_FAILURE);
    }

    hm_remove(e);
#endif
}

void print_memory_usage() {
#ifdef DEBUG_MEM
    printf("-----------------\n");
    printf("Peak memory usage: %zu bytes\n", peak_memory);
    printf("Current allocated memory: %zu bytes\n", allocated_memory);
    printf("Peak allocated count: %lld\n", peak_allocs);
    printf("Current allocated count: %lld\n", allocated_count);

    for (int i = 0; i < hm_num_buckets; i++) {
        entry* current = hm_buckets[i];
        while (current != NULL) {
            printf("  Entry: %p, size: %zu\n", current->ptr, current->size);
            current = current->next;
        }
    }
#else
    printf("Memory usage tracking is disabled.\n");
#endif
}