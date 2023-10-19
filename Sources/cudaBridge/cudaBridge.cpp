#include "../../External/Optix/7.7.0/include/optix_function_table_definition.h"

#include "OptixRenderer.h"
#include <fstream>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

void contextLogCallback(unsigned int level, const char *tag, const char *message, void *)
{
      fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
}

TriangleMesh triangleMesh;
Model model;

void optixAddTriangle(
	float ax, float ay, float az,
	float bx, float by, float bz,
	float cx, float cy, float cz)
{
	try {
		triangleMesh.addTriangle(ax, ay, az, bx, by, bz, cx, cy, cz);
	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}
}

OptixRenderer* optixRenderer;
std::vector<uint32_t> pixels;
std::vector<vec3f> vertices;
std::vector<vec3f> normals;
std::vector<int> intersected;
std::vector<int> primID;

void optixSetup() {
	try {
		model.meshes.push_back(&triangleMesh);
		optixRenderer = new OptixRenderer(&model);
		vec2i newSize {32, 32};
		optixRenderer->resize(newSize);
		pixels.resize(newSize.x * newSize.y);
		vertices.resize(newSize.x * newSize.y);
		normals.resize(newSize.x * newSize.y);
		intersected.resize(1);
		primID.resize(1);
	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}
}

void optixIntersect(
		float ox, float oy, float oz,
		float dx, float dy, float dz,
		float& tHit,
		float& px, float& py, float& pz,
		float& nx, float& ny, float& nz,
		int& didIntersect,
		int& didPrimID
		) {
	try {
		vec3f from = vec3f(ox, oy, oz);
		vec3f dir = vec3f(dx, dy, dz);

		Camera camera = {
			from,
			dir,
			tHit
		};
		optixRenderer->setCamera(camera);
		optixRenderer->render();

		optixRenderer->downloadPixels(pixels.data(), vertices.data(), normals.data(), intersected.data(), primID.data());
		auto& vertex = vertices[0];
		auto& normal = normals[0];
		px = vertex.x;
		py = vertex.y;
		pz = vertex.z;
		nx = normal.x;
		ny = normal.y;
		nz = normal.z;
		didIntersect = intersected[0];
		didPrimID = primID[0];
	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}
}

#ifdef __cplusplus
}
#endif

