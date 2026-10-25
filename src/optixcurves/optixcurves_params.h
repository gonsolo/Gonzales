// The launch parameters of optixcurves.cu, shared with optixcurves.cpp.
#pragma once
#include <optix_types.h>

struct OxcParams {
    OptixTraversableHandle handle;
    const float *rays;          // 8 floats a ray: origin, t_min (unused), direction, t_max
    unsigned int *results;      // closest: the RT results (8 words a ray), curve hits written in place
    const int *refs;            // per segment: curve, PrimId.id2, material
    const void *curves;         // the Curve array
    const int *list;            // shadow: ray indices, slot 0 from the front, slot 1 from the back
    const int *counts;          // shadow: rays in each slot's list
    unsigned char *blocked;     // shadow: per ray index, 1 when a curve is in the way
    int cap;
    int slot;
    float tmin;
};
