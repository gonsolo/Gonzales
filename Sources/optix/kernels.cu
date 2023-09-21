#include <stdio.h>
#include <cstdint>
#include "optix_device.h"
#include "LaunchParameters.h"

extern "C" __constant__ LaunchParameters launchParameters;

extern "C" __global__ void __closesthit__radiance() {}
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __miss__radiance() {}

__device__ void greet() {
	printf("Render frame kernel!\n");
}

extern "C" __global__ void __raygen__renderFrame() {
	const int x = optixGetLaunchIndex().x;
	const int y = optixGetLaunchIndex().y;
	//if (x == 0 && y  == 0) { greet(); }
	const uint8_t r = 255 * x / launchParameters.width;
	const uint8_t g = 255 * y / launchParameters.height;
	const uint8_t b = 0;
	const uint8_t a = 255;
	const int components = 4;
	const int index = y * launchParameters.width * components + x * components;
	uint8_t* p = (uint8_t*)launchParameters.pointerToPixels;
	p[index + 0] = r;
	p[index + 1] = g;
	p[index + 2] = b;
	p[index + 3] = a;


	// Not used yet
	float3 origin = { 0, 0, 0 };
	float3 direction = { 0, 0, 1 };
	float tMin = 0.f;
	float tMax = 1e20f;
	float rayTime = 0.f;
	int offset = 0;
	int stride = 1;
	int missIndex = 0;
	uint32_t u0 = 0;
	uint32_t u1 = 0;

	optixTrace(
		launchParameters.traversable,
                origin,
                direction,
                tMin,
                tMax,
                rayTime,
		OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_DISABLE_ANYHIT, // OPTIX_RAY_FLAG_NONE,
                offset,
                stride,
                missIndex,
                u0, u1 );
}
