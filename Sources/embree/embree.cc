#include <iostream>
#include <embree3/rtcore.h>

void initEmbree() {
        std::cout << "embree" << std::endl;
        RTCDevice device = rtcNewDevice(NULL);
}

