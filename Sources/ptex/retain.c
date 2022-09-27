// Print the retain count of all objects if above a certain limit
// Since swift_retain shows up in perf runs it is good to know which objects
// are responsible for that.

#if 0

#include <stddef.h>
#include <stdio.h>

extern void *(*_swift_retain)(void *);
static void *(*_old_swift_retain)(void*);
const char *swift_getTypeName(void *classObject, _Bool qualified);
size_t swift_retainCount(void *);

static void *swift_retain_hook(void *object) {
        int limit = 70;
        int count = swift_retainCount(object);
        if (count > limit) {
                void *isa = *(void**)object;
                const char *className = swift_getTypeName(isa, 1);
                fprintf(stderr, "%s at %p has more than %i retains!\n", className, object, count);
        }
  return _old_swift_retain(object);
}

__attribute__((constructor))
static void hook_swift_retain() {
  _old_swift_retain = _swift_retain;
  _swift_retain = swift_retain_hook;
}

#endif
