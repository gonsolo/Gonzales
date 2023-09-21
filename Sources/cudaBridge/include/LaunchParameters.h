#pragma once

#include "../../../External/Optix/7.7.0/include/optix.h"


struct LaunchParameters {
	int frameId { 0 };
	int width { 0 };
	int height { 0 };
	void *pointerToPixels;
	OptixTraversableHandle traversable;
};

