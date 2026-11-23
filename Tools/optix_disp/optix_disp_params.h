#pragma once
#include "disp_march.h"
#include <optix_types.h>

namespace gz {
struct OdCam {   // disp_load.h's Cam, device side
        V3 eye, fwd, right, up;
        float tan_h;
        int w, h;
        HD V3 dir(int x, int y) const {
                float px = (2.f * (float(x) + 0.5f) / float(w) - 1.f) * tan_h;
                float py = (1.f - 2.f * (float(y) + 0.5f) / float(h)) * tan_h * float(h) / float(w);
                V3 d = fwd + right * px + up * py;
                return d * (1.f / sqrtf(dot(d, d)));
        }
};
struct OdHit {
        float t;
        int32_t patch;
        float s, u;
};
} // namespace gz

struct OdParams {
        OptixTraversableHandle handle;
        gz::Surf s;
        gz::OdCam cam;
        gz::OdHit *hits;
        unsigned int *steps;
};
