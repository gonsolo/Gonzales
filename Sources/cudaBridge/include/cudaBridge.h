#pragma once

#include <cstdint>

#include "LaunchParameters.h"

#ifdef __cplusplus
extern "C" {
#endif
	void contextLogCallback(unsigned int level, const char *tag, const char *message, void *);
	void optixAddTriangle(float, float, float, float, float, float, float, float, float);
	void optixSetup();
	void optixIntersect(bool, float, float, float, float, float, float, float&, float&, float&, float&, float&, float&, float&, int&, int&);
#ifdef __cplusplus
}
#endif

