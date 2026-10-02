// Same symbols as rtcore.cpp for machines without the CUDA toolkit (the Mojo link line always pulls in -lrtcore).
#include "rtcore.h"
extern "C" void* rtcore_create(const char*, const uint8_t*, int64_t, uint64_t) { return nullptr; }
extern "C" int rtcore_trace(void*, uint64_t, uint64_t, int32_t, void*) { return 0; }
extern "C" void rtcore_destroy(void*) {}
