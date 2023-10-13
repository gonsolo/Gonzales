#pragma once

#include <cstdint>

#include "LaunchParameters.h"

#ifdef __cplusplus
extern "C" {
#endif
	void contextLogCallback(unsigned int level, const char *tag, const char *message, void *);
	void gonzoAdd(float, float, float, float, float, float, float, float, float);
	void gonzoSetup();
	void gonzoRender(bool, float, float, float, float, float, float, float&, float&, float&, float&, float&, float&, float&);
	void gonzoWrite();
#ifdef __cplusplus
}
#endif

