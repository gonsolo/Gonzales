// Mojo's allocator, routed to malloc so that ASan sees every Mojo allocation.
#include <stdlib.h>
void *KGEN_CompilerRT_AlignedAlloc(size_t align, size_t size) {
    void *p = 0;
    if (align < sizeof(void *)) align = sizeof(void *);
    if (posix_memalign(&p, align, size ? size : 1)) return 0;
    return p;
}
void KGEN_CompilerRT_AlignedFree(void *p) { free(p); }
