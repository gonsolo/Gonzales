#pragma once

#include <cstdint>
#include "../vec.h"

#ifdef __cplusplus
extern "C" {
#endif
	void contextLogCallback(unsigned int level, const char *tag, const char *message, void *);
	void optixAddTriangle(float, float, float, float, float, float, float, float, float);
	void optixSetup();
	void optixIntersect(
		vec3f from,
		float dx, float dy, float dz,
		float& tHit,
		float& px, float& py, float& pz,
		float& nx, float& ny, float& nz,
		int& didIntersect,
		int& didPrimID);
#ifdef __cplusplus
}
#endif

