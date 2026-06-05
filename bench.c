//
// Created by Tarcisio Zotelli Ferraz on 06/05/26.
//

#define _POSIX_C_SOURCE 199309L

#include <stdio.h>
#include <time.h>

#include "bench.h"

struct timespec start, end;

void bench_start(const char* label) {
    printf("-----------------\n");
    printf("Running benchmark %s\n", label);
    clock_gettime(CLOCK_MONOTONIC, &start);
}

void bench_end(const size_t N, const size_t M) {
    clock_gettime(CLOCK_MONOTONIC, &end);
    const unsigned long elapsed =
        (end.tv_sec - start.tv_sec) * 1000000000L
      + (end.tv_nsec - start.tv_nsec);

    printf("\telapsed ");
    if (elapsed > 1000000000) {
        printf("%lus\n", elapsed/1000000000L);
    }
    else if (elapsed > 1000000) {
        printf("%lums\n", elapsed/1000000L);
    }
    else if (elapsed > 1000) {
        printf("%luus\n", elapsed/1000L);
    }
    else {
        printf("%luns\n", elapsed);
    }
    const size_t ops = N*M;
    printf("\tOPs: %lu\n", ops);
    printf("\tOP/ns: %lu\n", elapsed/ops);
    printf("-----------------\n");
}
