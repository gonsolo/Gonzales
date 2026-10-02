// Same symbols as rtcore.cpp for machines without the CUDA toolkit (the Mojo link line always pulls in -lrtcore).
#include "rtcore.h"
extern "C" void* rtcore_create(const char*, const uint8_t*, int64_t, uint64_t) { return nullptr; }
extern "C" int rtcore_trace(void*, uint64_t, uint64_t, int32_t, void*) { return 0; }
extern "C" void rtcore_destroy(void*) {}
extern "C" int rtcore_set_meshes(void*, const int32_t*, int32_t) { return 0; }
extern "C" int rtcore_trace_interop(void*, uint64_t, uint64_t, int32_t, void*) { return 0; }
extern "C" void rtcore_set_active(void*) {}
extern "C" void* rtcore_active(void) { return nullptr; }
extern "C" void rtcore_set_shadow(int) {}
extern "C" int rtcore_set_domains(void*, int32_t, const int32_t*, const int32_t*, const int32_t*, const int32_t*, const int32_t*, int32_t) { return 0; }
extern "C" void* rtcore_create_scene(const char*, int32_t, const uint8_t* const*, const int64_t*, const uint64_t*) { return nullptr; }
extern "C" int rtcore_shadow_enabled(void) { return 0; }
extern "C" void rtcore_set_alpha(int) {}
extern "C" int rtcore_alpha_enabled(void) { return 0; }
