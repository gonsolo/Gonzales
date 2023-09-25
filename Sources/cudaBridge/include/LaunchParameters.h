#pragma once

#include "../../../External/Optix/7.7.0/include/optix.h"

#include <array>

struct vec3f {
	float x, y, z;
};

struct vec2i {
	int x, y;
};

struct Camera {
	vec3f position;
	vec3f direction;
	vec2i pixel;
};

struct LaunchParameters {
	int frameId { 0 };
	int width { 0 };
	int height { 0 };
	void *pointerToPixels;
	OptixTraversableHandle traversable;

	Camera camera;
	//std::array<vec3f, 32 * 32> cameraPositions;
	//std::array<vec3f, 32 * 32> cameraDirections;
};

