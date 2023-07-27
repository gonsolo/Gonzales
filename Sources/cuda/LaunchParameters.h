#pragma once

#include <cstdint>
#include <iostream>

struct LaunchParameters {
	int frameId { 0 };
	//uint32_t *pointerToPixels;
	void *pointerToPixels;
};

void* colorPointer;
