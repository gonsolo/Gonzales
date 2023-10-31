#pragma once

#include "../vec.h"
#include <cstdint>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

using VectorVec3f = std::vector<vec3f>;
using VectorFloat = std::vector<float>;
using VectorInt32 = std::vector<int32_t>;
using VectorBool = std::vector<bool>;

void contextLogCallback(unsigned int level, const char *tag, const char *message, void *);
void optixAddTriangle(vec3f a, vec3f b, vec3f c);
void optixSetup();
void optixIntersect(vec3f from, vec3f dir, float &tHit, vec3f &p, vec3f &n, int &didIntersect, int &didPrimID,
                    float &didTMax, bool skip);
void optixIntersectVec(
	const VectorVec3f& from,
	const VectorVec3f& dir,
	VectorFloat &tHit,
	VectorVec3f &p,
	VectorVec3f &n,
	VectorInt32 &didIntersect,
	VectorInt32  &didPrimID,
        VectorFloat &didTMax,
	VectorBool skip
);
#ifdef __cplusplus
}
#endif
