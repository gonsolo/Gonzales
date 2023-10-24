#pragma once

#include "../vec.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif
void contextLogCallback(unsigned int level, const char *tag, const char *message, void *);
void optixAddTriangle(vec3f a, vec3f b, vec3f c);
void optixSetup();
void optixIntersect(vec3f from, vec3f dir, float &tHit, vec3f &p, vec3f &n, int &didIntersect,
                    int &didPrimID, float &didTMax);
#ifdef __cplusplus
}
#endif
