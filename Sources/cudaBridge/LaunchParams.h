#pragma once

#include "optix7.h"
#include "vec.h"

struct TriangleMeshSBTData {
        vec3f color;
        vec3f *vertex;
        vec3f *normal;
        vec2f *texcoord;
        vec3i *index;
        bool hasTexture;
        cudaTextureObject_t texture;
};

struct LaunchParams {
        struct {
                uint32_t *colorBuffer;
                vec2i size;

                vec3f *outVertexBuffer;
                vec3f *outNormalBuffer;
                int *intersected;
                int *primID;
                float *tMax;
        } frame;

        struct {
                vec3f *position;
                vec3f *rayDirection;
                float *tHit;
        } camera;

        OptixTraversableHandle traversable;
};
