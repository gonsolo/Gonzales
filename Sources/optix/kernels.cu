#include <stdio.h>
#include <cstdint>
#include "optix_device.h"
#include "LaunchParameters.h"

extern "C" __constant__ LaunchParameters launchParameters;

extern "C" __global__ void __closesthit__radiance() {}
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __miss__radiance() {}
extern "C" __global__ void __raygen__renderFrame() {
	const int x = optixGetLaunchIndex().x;
	const int y = optixGetLaunchIndex().y;
	//if (x == 0 && y  == 0) {
	//	printf("Render frame kernel, x, y: %i %i!\n", x, y);
	//}

	const uint8_t r = x * 16;
	const uint8_t g = 0;
	const uint8_t b = 0;
	const uint8_t a = 255;
	const int components = 4;
	const int index = y * launchParameters.width * components + x * components;
	uint8_t* p = (uint8_t*)launchParameters.pointerToPixels;
	p[index + 0] = r;
	p[index + 1] = g;
	p[index + 2] = b;
	p[index + 3] = a;
}
