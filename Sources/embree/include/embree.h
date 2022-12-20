#include <cstdint>
#include <embree3/rtcore.h>

void embreeSetDevice(RTCDevice device);
void embreeSetScene(RTCScene scene);
bool embreeIntersect3(
                //float ox, float oy, float oz,
                //float dx, float dy, float dz,
                //float tnear, float tfar,
                float& nx, float& ny, float& nz,
                float& tout,
                uint32_t& geomID,
                RTCRayHit rayhit,
                RTCIntersectContext context
);
