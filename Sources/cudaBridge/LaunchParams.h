#pragma once

#include "gdt/math/vec.h"
#include "optix7.h"

struct TriangleMeshSBTData {
        gdt::vec3f color;
        gdt::vec3f *vertex;
        gdt::vec3f *normal;
        gdt::vec2f *texcoord;
        gdt::vec3i *index;
        bool hasTexture;
        cudaTextureObject_t texture;
};

struct LaunchParams {
        struct {
                uint32_t *colorBuffer;
                gdt::vec2i size;

                gdt::vec3f *outVertexBuffer;
                gdt::vec3f *outNormalBuffer;
                int *intersected;
                int *primID;
        } frame;

        struct {
                gdt::vec3f position;
                gdt::vec3f rayDirection;
                float tHit;
        } camera;

        OptixTraversableHandle traversable;
};
