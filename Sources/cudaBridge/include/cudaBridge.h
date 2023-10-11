#pragma once

#include <cstdint>

#include "LaunchParameters.h"

#ifdef __cplusplus
extern "C" {
#endif
	void contextLogCallback(unsigned int level, const char *tag, const char *message, void *);
	void gonzoBla(float, float, float, float, float, float, float, float, float);
#ifdef __cplusplus
}
#endif

