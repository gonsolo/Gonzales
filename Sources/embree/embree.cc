#include <iostream>
#include <limits>
#include <string_view>
#include <embree3/rtcore.h>

RTCDevice device;
RTCScene scene;

void embreeError(std::string_view message) {
        std::cerr << "Embree error: " << message << std::endl;
        exit(EXIT_FAILURE);
}

void embreeSetDevice(RTCDevice d) {
        device = d;
}

void embreeSetScene(RTCScene s) {
        scene = s;
}

bool embreeIntersect(
                float ox, float oy, float oz,
                float dx, float dy, float dz,
                float tnear, float tfar,
                float& nx, float& ny, float& nz,
                float& tout,
                uint32_t& geomID
                ) {
        RTCRayHit rayhit; 
        rayhit.ray.org_x = ox; rayhit.ray.org_y = oy; rayhit.ray.org_z = oz;
        rayhit.ray.dir_x = dx; rayhit.ray.dir_y = dy; rayhit.ray.dir_z = dz;
        rayhit.ray.tnear = tnear;
        rayhit.ray.tfar = tfar;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;

        RTCIntersectContext context;
        rtcInitIntersectContext(&context);

        rtcIntersect1(scene, &context, &rayhit);

        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
                tout = rayhit.ray.tfar;
                geomID = rayhit.hit.geomID;
                nx = rayhit.hit.Ng_x;
                ny = rayhit.hit.Ng_y;
                nz = rayhit.hit.Ng_z;
                return true;
        } else {
                return false;
        }
}
