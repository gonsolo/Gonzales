// This just defines the function table.
// Without this there are link errors.
#include "../../External/Optix/7.7.0/include/optix_function_table_definition.h"

#include "SampleRenderer.h"
#include <fstream>

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

void contextLogCallback(unsigned int level, const char *tag, const char *message, void *)
{
      fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
}

osc::TriangleMesh triangleMesh;
osc::Model model;

void gonzoAdd(
	float ax, float ay, float az,
	float bx, float by, float bz,
	float cx, float cy, float cz)
{
	//puts("gonzoAdd!\n");
	try {
		//model.addCube(osc::vec3f(0.f,-1.5f,0.f),osc::vec3f(10.f,.1f,10.f));
		//model.addCube(osc::vec3f(0.f,0.f,0.f),osc::vec3f(2.f,2.f,2.f));

		triangleMesh.addTriangle(ax, ay, az, bx, by, bz, cx, cy, cz);
	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}
}

osc::SampleRenderer* sampleRenderer;
std::vector<uint32_t> pixels;
std::vector<gdt::vec3f> vertices;
std::vector<gdt::vec3f> normals;
std::vector<int> intersected;
std::vector<int> primID;

void gonzoSetup() {
	puts("gonzoSetup!\n");
	try {
		model.meshes.push_back(&triangleMesh);
		sampleRenderer = new osc::SampleRenderer(&model);
		osc::vec2i newSize {32, 32};
		sampleRenderer->resize(newSize);
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
		bool useRay,
		float ox, float oy, float oz,
		float dx, float dy, float dz,
		float& tHit,
		float& px, float& py, float& pz,
		float& nx, float& ny, float& nz,
		int& didIntersect,
		int& didPrimID
		) {
	try {
		osc::Camera camera = {
			/*from*/osc::vec3f(-10.f,2.f,-12.f),
                        /* at */osc::vec3f(0.f,0.f,0.f),
                        /* up */osc::vec3f(0.f,1.f,0.f),
			/* useRay */false,
			/* dir */osc::vec3f(0.f, 0.f, 0.f),
			/* tHit */1e20f
		};
		if (useRay) {
			camera.from = osc::vec3f(ox, oy, oz);
			camera.useRay = true;
			camera.rayDirection = osc::vec3f(dx, dy, dz);
			camera.tHit = tHit;
			//printf("  rayDirection: %f %f %f!\n", dx, dy, dz);
		}
		sampleRenderer->setCamera(camera);
		sampleRenderer->render();

		sampleRenderer->downloadPixels(pixels.data(), vertices.data(), normals.data(), intersected.data(), primID.data());
		auto& vertex = vertices[0];
		auto& normal = normals[0];
		//std::cout << "vertex: " << vertex << std::endl;
		px = vertex.x;
		py = vertex.y;
		pz = vertex.z;
		nx = normal.x;
		ny = normal.y;
		nz = normal.z;
		didIntersect = intersected[0];
		didPrimID = primID[0];
		//std::cout << didIntersect << std::endl;
	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}
}

#ifdef __cplusplus
}
#endif

