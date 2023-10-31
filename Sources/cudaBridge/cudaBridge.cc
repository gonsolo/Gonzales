#include "../../External/Optix/7.7.0/include/optix_function_table_definition.h"

#include "OptixRenderer.h"
#include "include/cudaBridge.h"
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
VectorUInt32 pixels;
VectorVec3f vertices;
VectorVec3f normals;
VectorInt32 intersected;
VectorInt32 primID;
VectorFloat tMax;

void optixSetup() {
        try {
		model.meshes.push_back(&triangleMesh);
		optixRenderer = new OptixRenderer(&model);
		vec2i newSize{1, 1};
		optixRenderer->resize(newSize);
		auto sizeVec = newSize.x * newSize.y;
		pixels.resize(sizeVec);
		vertices.resize(sizeVec);
		normals.resize(sizeVec);
		intersected.resize(sizeVec);
		primID.resize(sizeVec);
		tMax.resize(sizeVec);
        } catch (std::runtime_error &e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
        }
}

void optixIntersectVec(
	const VectorVec3f& from,
	const VectorVec3f& dir,
	VectorFloat &tHit,
	VectorVec3f &p,
	VectorVec3f &n,
	VectorInt32 &didIntersect,
	VectorInt32 &didPrimID,
        VectorFloat &didTMax,
	VectorBool skip
) {
	// ignore skip for now
        try {
                OptixCamera camera = {from, dir, tHit};
//                optixRenderer->setCamera(camera);
//                optixRenderer->render();
//                optixRenderer->downloadPixels(pixels.data(), vertices.data(), normals.data(),
//                                              intersected.data(), primID.data(), tMax.data());
//                p = vertices[0];
//                n = normals[0];
//                didIntersect = intersected[0];
//                didPrimID = primID[0];
//                didTMax = tMax[0];
        } catch (std::runtime_error &e) {
                std::cout << "FATAL ERROR: " << e.what() << std::endl;
                exit(1);
        }

	//
	for (int i = 0; i < from.size(); i++) {
		optixIntersect(
				from[i],
				dir[i],
				tHit[i],
				p[i],
				n[i],
				didIntersect[i],
				didPrimID[i],
				didTMax[i],
				skip[i]);
	}
}

void optixIntersect(
	vec3f from,
	vec3f dir,
	float &tHit,
	vec3f &p,
	vec3f &n,
	int &didIntersect,
	int &didPrimID,
        float &didTMax,
	bool skip
) {
	if (skip) {
		return;
	}
        try {
		VectorVec3f fromVec = { from };
		VectorVec3f dirVec = { dir };
		VectorFloat tHitVec = { tHit };
                OptixCamera camera = {fromVec, dirVec, tHitVec};
                optixRenderer->setCamera(camera);
                optixRenderer->render();
                optixRenderer->downloadPixels(pixels.data(), vertices.data(), normals.data(),
                                              intersected.data(), primID.data(), tMax.data());
                p = vertices[0];
                n = normals[0];
                didIntersect = intersected[0];
                didPrimID = primID[0];
                didTMax = tMax[0];
        } catch (std::runtime_error &e) {
                std::cout << "FATAL ERROR: " << e.what() << std::endl;
                exit(1);
        }
}

#ifdef __cplusplus
}
#endif
