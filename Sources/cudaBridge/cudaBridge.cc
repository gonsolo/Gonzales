#include "../../External/Optix/7.7.0/include/optix_function_table_definition.h"

#include "OptixRenderer.h"
#include <fstream>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

void contextLogCallback(unsigned int level, const char *tag, const char *message, void *) {
        fprintf(stderr, "[%2d][%12s]: %s\n", (int)level, tag, message);
}

TriangleMesh triangleMesh;
Model model;

void optixAddTriangle(vec3f a, vec3f b, vec3f c) {
        try {
                triangleMesh.addTriangle(a, b, c);
        } catch (std::runtime_error &e) {
                std::cout << "FATAL ERROR: " << e.what() << std::endl;
                exit(1);
        }
}

OptixRenderer *optixRenderer;
std::vector<uint32_t> pixels;
std::vector<vec3f> vertices;
std::vector<vec3f> normals;
std::vector<int> intersected;
std::vector<int> primID;

void optixSetup() {
        try {
                model.meshes.push_back(&triangleMesh);
                optixRenderer = new OptixRenderer(&model);
                vec2i newSize{1, 1};
                optixRenderer->resize(newSize);
                pixels.resize(newSize.x * newSize.y);
                vertices.resize(newSize.x * newSize.y);
                normals.resize(newSize.x * newSize.y);
                intersected.resize(1);
                primID.resize(1);
        } catch (std::runtime_error &e) {
                std::cout << "FATAL ERROR: " << e.what() << std::endl;
                exit(1);
        }
}

void optixIntersect(
		vec3f from,
		vec3f dir,
		float &tHit,
		vec3f& p,
		vec3f& n,
		int &didIntersect,
                int &didPrimID) {
        try {
                OptixCamera camera = {from, dir, tHit};
                optixRenderer->setCamera(camera);
                optixRenderer->render();

                optixRenderer->downloadPixels(pixels.data(), vertices.data(), normals.data(),
                                              intersected.data(), primID.data());
                auto &vertex = vertices[0];
                auto &normal = normals[0];
                p.x = vertex.x;
                p.y = vertex.y;
                p.z = vertex.z;
                n.x = normal.x;
                n.y = normal.y;
                n.z = normal.z;
                didIntersect = intersected[0];
                didPrimID = primID[0];
        } catch (std::runtime_error &e) {
                std::cout << "FATAL ERROR: " << e.what() << std::endl;
                exit(1);
        }
}

#ifdef __cplusplus
}
#endif
