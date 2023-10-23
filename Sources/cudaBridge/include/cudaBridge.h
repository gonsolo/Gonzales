#pragma once

#include <cstdint>
#include "../vec.h"

#ifdef __cplusplus
extern "C" {
#endif
	void contextLogCallback(unsigned int level, const char *tag, const char *message, void *);
	void optixAddTriangle(vec3f a, vec3f b, vec3f c);
	void optixSetup();
	void optixIntersect(
		vec3f from,
		vec3f dir,
		float& tHit,
		float& px, float& py, float& pz,
		float& nx, float& ny, float& nz,
		int& didIntersect,
		int& didPrimID);
#ifdef __cplusplus
}
#endif

