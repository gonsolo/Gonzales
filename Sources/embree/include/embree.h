#include <cstdint>
#include <embree3/rtcore.h>


void embreeCommit();
void embreeSetDevice(RTCDevice device);
void embreeSetScene(RTCScene scene);
void embreeDeinit();
void embreeGeometry(
                float ax, float ay, float az,
                float bx, float by, float bz,
                float cx, float cy, float cz);
bool embreeIntersect(
                float ox, float oy, float oz,
                float dx, float dy, float dz,
                float tnear, float tfar,
                float& nx, float& ny, float& nz,
                float& tout,
                uint32_t& geomID);
