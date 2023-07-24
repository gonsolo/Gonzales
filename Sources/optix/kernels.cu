#include <stdio.h>
#include "optix_device.h"
#include "LaunchParameters.h"

extern "C" __constant__ LaunchParameters launchParameters;

extern "C" __global__ void __closesthit__radiance() {}
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __miss__radiance() {}
extern "C" __global__ void __raygen__renderFrame() {
	const int x = optixGetLaunchIndex().x;
	const int y = optixGetLaunchIndex().y;
	if (
		x == 0 && y  == 0
	) {
		printf("Render frame kernel, frame id: %i!\n",
			launchParameters.frameId);
	}

	const uint32_t rgba = 0xffffffff;
	//const uint32_t index = y * 16 + x;
	const uint32_t index = 0;
	uint32_t* p = (uint32_t*)launchParameters.pointerToPixels;
	//launchParameters.pointerToPixels[index] = rgba;
	//p[index] = rgba;
}
