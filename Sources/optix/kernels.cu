#include <stdio.h>
#include "optix_device.h"
#include "LaunchParameters.h"

extern "C" __constant__ LaunchParameters launchParameters;

extern "C" __global__ void __closesthit__radiance() {}
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __miss__radiance() {}
extern "C" __global__ void __raygen__renderFrame() {
	if (
		//launchParameters.frameID == 0 &&
		optixGetLaunchIndex().x == 0 &&
		optixGetLaunchIndex().y == 0
	) {
		printf("Render frame kernel, frame id: %i!\n",
			launchParameters.frameId);
	}
}
