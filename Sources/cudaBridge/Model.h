#pragma once

#include "include/vec.h"
#include <vector>

struct TriangleMesh {
        std::vector<vec3f> vertex;
        std::vector<vec3f> normal;
        std::vector<vec2f> texcoord;
        std::vector<vec3i> index;

        vec3f diffuse;
        int diffuseTextureID{-1};

        void addTriangle(vec3f a, vec3f b, vec3f c) {
                int firstVertexID = (int)vertex.size();
                vertex.push_back(a);
                vertex.push_back(b);
                vertex.push_back(c);
                int indices[] = {0, 1, 2};
                index.push_back(firstVertexID + vec3i(indices[0], indices[1], indices[2]));
        }
};

struct Model {
        std::vector<TriangleMesh *> meshes;
};
