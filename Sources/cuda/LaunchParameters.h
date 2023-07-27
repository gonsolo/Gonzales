#pragma once

#include <cstdint>

struct LaunchParameters {
	int frameId { 0 };
	int width { 0 };
	int height { 0 };
	void *pointerToPixels;
};

