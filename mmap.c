//
// Created by tzf on 5/6/26.
//
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "mmap.h"

const u8* mopen(const char* path, size_t* dest_size) {
    // 1. Open the file
    const int fd = open(path, O_RDONLY);
    if (fd == -1) {
        printf("open %s failed\n", path);
        return nullptr;
    }

    // 2. Get file size
    struct stat st;
    fstat(fd, &st);

    // 3. Map the file into memory
    // NULL: Let kernel choose address
    // st.st_size: Length of mapping
    // PROT_READ: Read-only access
    // MAP_PRIVATE: Changes are not visible to other processes or the file
    const u8* ptr = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

    if (ptr == MAP_FAILED) {
        printf("mmap failed\n");
        close(fd);
        return NULL;
    }

    *dest_size = st.st_size;

    return ptr;
}