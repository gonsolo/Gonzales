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
	if (
		x == 0 && y  == 0
	) {
		printf("Render frame kernel, frame id: %i!\n",
			launchParameters.frameId);
	}

	//const uint32_t rgba = 666;
	const uint8_t r = 255;
	const uint8_t g = 128;
	const uint8_t b = 10;
	const uint8_t a = 255;
	const int width = 16;
	const int components = 4;
	const uint8_t index = y * width * components + x * components;
	const uint8_t indexRed = index + 0;
	const uint8_t indexGreen = index + 1;
	const uint8_t indexBlue = index + 2;
	const uint8_t indexAlpha = index + 3;
	uint8_t* p = (uint8_t*)launchParameters.pointerToPixels;
	p[indexRed] = r;
	p[indexGreen] = g;
	p[indexBlue] = b;
	p[indexAlpha] = a;
}
