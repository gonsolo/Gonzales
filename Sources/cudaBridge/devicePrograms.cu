#include <optix_device.h>
#include <cuda_runtime.h>

#include "LaunchParams.h"
#include "vec.h"

extern "C" __constant__ LaunchParams optixLaunchParams;

enum { SURFACE_RAY_TYPE=0, RAY_TYPE_COUNT };

static __forceinline__ __device__
void *unpackPointer( uint32_t i0, uint32_t i1 )
{
  const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
  void*           ptr = reinterpret_cast<void*>( uptr ); 
  return ptr;
}

static __forceinline__ __device__
void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
{
  const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
  i0 = uptr >> 32;
  i1 = uptr & 0x00000000ffffffff;
}

template<typename T>
static __forceinline__ __device__ T *getPRD()
{ 
  const uint32_t u0 = optixGetPayload_0();
  const uint32_t u1 = optixGetPayload_1();
  return reinterpret_cast<T*>( unpackPointer( u0, u1 ) );
}

struct PerRayData {
	vec3f intersectionPoint;
	vec3f intersectionNormal;
  int intersected = 0;
  int primID = -1;
};

extern "C" __global__ void __closesthit__radiance()
{
  const TriangleMeshSBTData &sbtData
    = *(const TriangleMeshSBTData*)optixGetSbtDataPointer();
  
  const int   primID = optixGetPrimitiveIndex();
  const vec3i index  = sbtData.index[primID];
  const float u = optixGetTriangleBarycentrics().x;
  const float v = optixGetTriangleBarycentrics().y;

  vec3f N;
  if (sbtData.normal) {
    N = (1.f-u-v) * sbtData.normal[index.x]
      +         u * sbtData.normal[index.y]
      +         v * sbtData.normal[index.z];
  } else {
    const vec3f &A     = sbtData.vertex[index.x];
    const vec3f &B     = sbtData.vertex[index.y];
    const vec3f &C     = sbtData.vertex[index.z];
    N                  = normalize(cross(B-A,C-A));
  }
  N = normalize(N);

  const vec3f &A     = sbtData.vertex[index.x];
  const vec3f &B     = sbtData.vertex[index.y];
  const vec3f &C     = sbtData.vertex[index.z];
  vec3f P = (1.f-u-v) * A + u * B + v * C;
  PerRayData &prd = *(PerRayData*)getPRD<PerRayData>();
  prd.intersectionPoint = P;
  prd.intersectionNormal = N;
  prd.intersected = 1;
  prd.primID = primID;
}

extern "C" __global__ void __anyhit__radiance() {}

extern "C" __global__ void __miss__radiance()
{
  PerRayData &prd = *(PerRayData*)getPRD<PerRayData>();
  prd.intersectionPoint = vec3f(1.f);
  prd.intersected = 0;
}

extern "C" __global__ void __raygen__renderFrame()
{
  const int ix = optixGetLaunchIndex().x;
  const int iy = optixGetLaunchIndex().y;

  const auto &camera = optixLaunchParams.camera;

  PerRayData perRayData = { vec3f(0.f), vec3f(0.f), false, -1 };

  uint32_t u0, u1;
  packPointer( &perRayData, u0, u1 );

  float3 rayDir;
  rayDir.x = camera.rayDirection.x;
  rayDir.y = camera.rayDirection.y;
  rayDir.z = camera.rayDirection.z;

  float tmax = camera.tHit;

  float3 position;
  position.x = camera.position.x;
  position.y = camera.position.y;
  position.z = camera.position.z;

  optixTrace(optixLaunchParams.traversable,
             position,
             rayDir,
             0.f,    // tmin
             tmax,
             0.0f,   // rayTime
             OptixVisibilityMask( 255 ),
             OPTIX_RAY_FLAG_DISABLE_ANYHIT,//OPTIX_RAY_FLAG_NONE,
             SURFACE_RAY_TYPE,             // SBT offset
             RAY_TYPE_COUNT,               // SBT stride
             SURFACE_RAY_TYPE,             // missSBTIndex 
             u0, u1 );

  const int r = int(255.99f*perRayData.intersectionPoint.x);
  const int g = int(255.99f*perRayData.intersectionPoint.y);
  const int b = int(255.99f*perRayData.intersectionPoint.z);

  const uint32_t rgba = 0xff000000
    | (r<<0) | (g<<8) | (b<<16);

  const uint32_t fbIndex = ix+iy*optixLaunchParams.frame.size.x;
  optixLaunchParams.frame.colorBuffer[fbIndex] = rgba;
  optixLaunchParams.frame.outVertexBuffer[fbIndex] = perRayData.intersectionPoint;
  optixLaunchParams.frame.outNormalBuffer[fbIndex] = perRayData.intersectionNormal;
  optixLaunchParams.frame.intersected[0] = perRayData.intersected;
  optixLaunchParams.frame.primID[0] = perRayData.primID;
}
