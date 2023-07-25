#include <cstdint>
#include <iostream>

struct LaunchParameters {
	int frameId { 0 };
	//uint32_t *pointerToPixels;
	void *pointerToPixels;
};

void* colorPointer;

CUdeviceptr buildLaunch() {
	LaunchParameters launchParameters;

	auto colorError = cudaMalloc(&colorPointer, 16);
	std::cerr << colorError << std::endl;

	launchParameters.pointerToPixels = colorPointer;

	void* launchPointer;
	auto launchError = cudaMalloc(&launchPointer, sizeof(LaunchParameters));
	std::cerr << launchError << std::endl;

	auto uploadError = cudaMemcpy(launchPointer, &launchParameters, sizeof(launchParameters), cudaMemcpyHostToDevice);	
	std::cerr << uploadError << std::endl;


	return (CUdeviceptr)launchPointer;
}

uint32_t getColor() {
	uint32_t i;
	auto error = cudaMemcpy(&i, colorPointer, sizeof(uint32_t), cudaMemcpyDeviceToHost);
	std::cerr << error << std::endl;
	return i;
}

