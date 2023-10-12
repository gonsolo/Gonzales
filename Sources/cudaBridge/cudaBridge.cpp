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

osc::TriangleMesh model;

void gonzoAdd(
	float ax, float ay, float az,
	float bx, float by, float bz,
	float cx, float cy, float cz)
{
	puts("gonzoAdd!\n");
	try {
		//model.addCube(osc::vec3f(0.f,-1.5f,0.f),osc::vec3f(10.f,.1f,10.f));
		//model.addCube(osc::vec3f(0.f,0.f,0.f),osc::vec3f(2.f,2.f,2.f));
		model.addTriangle(ax, ay, az, bx, by, bz, cx, cy, cz);
	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}
}

osc::SampleRenderer* sampleRenderer;
std::vector<uint32_t> pixels;

void gonzoSetup() {
	puts("gonzoSetup!\n");
	try {
		osc::Camera camera = {
			/*from*/osc::vec3f(-10.f,2.f,-12.f),
                        /* at */osc::vec3f(0.f,0.f,0.f),
                        /* up */osc::vec3f(0.f,1.f,0.f) };
		//osc::SampleRenderer sampleRenderer(model);
		sampleRenderer = new osc::SampleRenderer(model);
		sampleRenderer->setCamera(camera);
		osc::vec2i newSize {32, 32};
		sampleRenderer->resize(newSize);
		pixels.resize(newSize.x * newSize.y);
	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}
}

void gonzoRender() {
	puts("gonzoRender!\n");
	try {
		sampleRenderer->render();

	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}
}

void gonzoWrite() {
	puts("gonzoWrite!\n");
	try {
		sampleRenderer->downloadPixels(pixels.data());
		std::ofstream bla("bla.ppm");
		bla << "P3" << std::endl;
		bla << "32 32 " << std::endl;
		bla << "255" << std::endl;
		for(auto i : pixels) {
			auto r = (i & 0x000000ff) >> 0;
			auto g = (i & 0x0000ff00) >> 8;
			auto b = (i & 0x00ff0000) >> 16;
			bla << r << " " << g << " " << b << std::endl;
		}
	} catch (std::runtime_error& e) {
		std::cout << "FATAL ERROR: " << e.what() << std::endl;
		exit(1);
	}
}

#ifdef __cplusplus
}
#endif

