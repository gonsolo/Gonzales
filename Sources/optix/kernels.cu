#include <stdio.h>
#include <cstdint>
#include "optix_device.h"
#include "LaunchParameters.h"

extern "C" __constant__ LaunchParameters launchParameters;

static __forceinline__ __device__
void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
	const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
	i0 = uptr >> 32;
	i1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__
void *unpackPointer( uint32_t i0, uint32_t i1 )
{
	const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
	void*           ptr = reinterpret_cast<void*>( uptr );
	return ptr;
}


template<typename T>
static __forceinline__ __device__ T *getPerRayData()
{
	const uint32_t u0 = optixGetPayload_0();
	const uint32_t u1 = optixGetPayload_1();
	return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}

extern "C" __global__ void __closesthit__radiance() {
	printf("closesthit");
	const int   primID = optixGetPrimitiveIndex();
	vec3f &perRayData = *(vec3f*)getPerRayData<vec3f>();
	perRayData = {1, 1, 1};
}

extern "C" __global__ void __anyhit__radiance() {
	printf("closesthit");
}

extern "C" __global__ void __miss__radiance() {
	//printf("miss");
	vec3f &perRayData = *(vec3f*)getPerRayData<vec3f>();
	perRayData = {0, 0, 0};
}

__device__ void greet() {
	printf("Render frame kernel!\n");
}

extern "C" __global__ void __raygen__renderFrame() {
	const int x = optixGetLaunchIndex().x;
	const int y = optixGetLaunchIndex().y;
	//if (x == 0 && y  == 0) { greet(); }
	uint8_t r = 255 * x / launchParameters.width;
	uint8_t g = 255 * y / launchParameters.height;
	uint8_t b = 0;
	const uint8_t a = 255;
	const int components = 4;
	const int index = y * launchParameters.width * components + x * components;
	uint8_t* p = (uint8_t*)launchParameters.pointerToPixels;
	//p[index + 0] = r;
	//p[index + 1] = g;
	//p[index + 2] = b;
	//p[index + 3] = a;

	vec3f perRayData;
	uint32_t u0 = 0;
	uint32_t u1 = 0;
	packPointer( &perRayData, u0, u1 );

	float3 origin = { 
		launchParameters.camera.position.x,
		launchParameters.camera.position.y,
		launchParameters.camera.position.z
	};

	float3 direction = {
		launchParameters.camera.direction.x,	
		launchParameters.camera.direction.y,
		launchParameters.camera.direction.z
	};
	if(x == launchParameters.camera.pixel.x && y == launchParameters.camera.pixel.y) {
		//printf("position: %f %f %f\n", launchParameters.camera.position.x, launchParameters.camera.position.y, launchParameters.camera.position.z);
		//printf("direction: %f %f %f\n", launchParameters.camera.direction.x, launchParameters.camera.direction.y, launchParameters.camera.direction.z);
	}

	float tMin = 0.f;
	float tMax = 1e20f;
	float rayTime = 0.f;
	int offset = 0;
	int stride = 1;
	int missIndex = 0;

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

	r = int(255.99f * perRayData.x);
	g = int(255.99f * perRayData.y);
	b = int(255.99f * perRayData.z);


	if(x == launchParameters.camera.pixel.x && y == launchParameters.camera.pixel.y) {
		//printf("%i %i %i %i\n", x, launchParameters.camera.pixel.x, y, launchParameters.camera.pixel.y);
		p[index + 0] = r;
		p[index + 1] = g;
		p[index + 2] = b;
		p[index + 3] = a;
	}
}
