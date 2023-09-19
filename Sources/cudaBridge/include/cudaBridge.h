#pragma once

#include <cstdint>

#include "LaunchParameters.h"

#ifdef __cplusplus
extern "C" {
#endif
	void contextLogCallback(unsigned int level, const char *tag, const char *message, void *);
#ifdef __cplusplus
}
#endif

//uint32_t triangleInputFlags[1] = { 0 };
