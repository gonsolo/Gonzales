#pragma once

#include <cstdint>

#include "LaunchParameters.h"

#ifdef __cplusplus
extern "C" {
#endif
	void contextLogCallback(unsigned int level, const char *tag, const char *message, void *);
	void gonzoAdd(float, float, float, float, float, float, float, float, float);
	void gonzoRender();
#ifdef __cplusplus
}
#endif

