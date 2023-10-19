#pragma once

#include "vec.h"
#include <vector>

struct TriangleMesh {
        std::vector<vec3f> vertex;
        std::vector<vec3f> normal;
        std::vector<vec2f> texcoord;
        std::vector<vec3i> index;

        vec3f diffuse;
        int diffuseTextureID{-1};

        void addTriangle(float ax, float ay, float az, float bx, float by, float bz, float cx, float cy,
                         float cz) {
                int firstVertexID = (int)vertex.size();
                vertex.push_back(vec3f(ax, ay, az));
                vertex.push_back(vec3f(bx, by, bz));
                vertex.push_back(vec3f(cx, cy, cz));

                int indices[] = {0, 1, 2};
                index.push_back(firstVertexID + vec3i(indices[0], indices[1], indices[2]));
        }
};

struct Model {
        std::vector<TriangleMesh *> meshes;
};
