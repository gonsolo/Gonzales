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

void embreeInit() {
        device = rtcNewDevice(NULL);
        if (!device) {
                embreeError("rtcNewDevice");
        }
        scene = rtcNewScene(device);
        if (!scene) {
                embreeError("rtNewScene");
        }
}

void embreeDeinit() {
        rtcReleaseScene(scene);
        rtcReleaseDevice(device);
}

void embreeGeometry(
                float ax, float ay, float az,
                float bx, float by, float bz,
                float cx, float cy, float cz
) {
        RTCGeometry geom = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
        if (!geom) {
                embreeError("rtcNewGeometry");
        }
        auto vb = (float*) rtcSetNewGeometryBuffer(
                        geom,
                        RTC_BUFFER_TYPE_VERTEX,
                        0,
                        RTC_FORMAT_FLOAT3,
                        3 * sizeof(float),
                        3);
        if (!vb) {
                embreeError("rtcSetNewGeometryBuffer");
        }
        vb[0] = ax; vb[1] = ay; vb[2] = az; // 1st vertex
        vb[3] = bx; vb[4] = by; vb[5] = bz; // 2nd vertex
        vb[6] = cx; vb[7] = cy; vb[8] = cz; // 3rd vertex

        auto ib = (unsigned*) rtcSetNewGeometryBuffer(
                        geom,
                        RTC_BUFFER_TYPE_INDEX,
                        0,
                        RTC_FORMAT_UINT3,
                        3 * sizeof(unsigned),
                        1);
        if (!ib) {
                embreeError("rtcSetNewGeometryBuffer");
        }
        ib[0] = 0; ib[1] = 1; ib[2] = 2;
        
        rtcCommitGeometry(geom);
        rtcAttachGeometry(scene, geom);
        rtcReleaseGeometry(geom);
}

void embreeCommit() {
        rtcCommitScene(scene);
}

bool embreeIntersect(
                float ox, float oy, float oz,
                float dx, float dy, float dz,
                float tnear, float tfar,
                float& nx, float& ny, float& nz,
                float& tout
) {
        RTCRayHit rayhit; 
        //rayhit.ray.org_x  = 0.f; rayhit.ray.org_y = 0.f; rayhit.ray.org_z = -1.f;
        //rayhit.ray.dir_x  = 0.f; rayhit.ray.dir_y = 0.f; rayhit.ray.dir_z =  1.f;
        //rayhit.ray.tnear  = 0.f;
        //rayhit.ray.tfar   = std::numeric_limits<float>::infinity();
        rayhit.ray.org_x = ox; rayhit.ray.org_y = oy; rayhit.ray.org_z = oz;
        rayhit.ray.dir_x = dx; rayhit.ray.dir_y = dy; rayhit.ray.dir_z = dz;
        rayhit.ray.tnear = tnear;
        rayhit.ray.tfar = tfar;
        rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
        
        RTCIntersectContext context;
        rtcInitIntersectContext(&context);
        
        rtcIntersect1(scene, &context, &rayhit);
        
        if (rayhit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
          //std::cout << "Intersection at t = " << rayhit.ray.tfar << std::endl;
          tout = rayhit.ray.tfar;
          nx = rayhit.hit.Ng_x;
          ny = rayhit.hit.Ng_y;
          nz = rayhit.hit.Ng_z;
          return true;
        } else {
          //std::cout << "No Intersection" << std::endl;
          return false;
        }
}
