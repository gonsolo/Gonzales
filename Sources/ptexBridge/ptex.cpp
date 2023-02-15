#include <Ptexture.h>
#include <cassert>
#include <iostream>
#include <string>

struct GonzalesPtexHandler : public PtexErrorHandler {
        void reportError(const char *error) { std::cerr << "Ptex error: " << error << std::endl; }
} handler;

namespace ptexBridge {
void initCache(long int memory);
void initTexture(const char *filename);
void evaluate(const char *filename, long int faceIndex, float u, float v, float rgb[3]);
}

#ifdef __cplusplus
extern "C" {
#endif

void initPtexCache(long int memory) { ptexBridge::initCache(memory); }

void initPtexTexture(const char *filename) { ptexBridge::initTexture(filename); }

void evaluatePtex(const char *filename, long int faceIndex, float u, float v, float rgb[3]) {
        ptexBridge::evaluate(filename, faceIndex, u, v, rgb);
}

#ifdef __cplusplus
}
#endif

namespace ptexBridge {

using namespace Ptex;

PtexCache *cache;
String error;

void initCache(long int memory) {
#ifdef __x86_64
        int maxFiles = 1024;
        size_t maxMemory = memory * 1024l * 1024l * 1024l; // 4GB
        bool premultiply = true;
        auto inputHandler = nullptr;
        cache = PtexCache::create(maxFiles, maxMemory, premultiply, inputHandler, &handler);
        if (!cache) {
                std::cerr << "No ptex cache" << std::endl;
        }
#endif
}

void initTexture(const char *filename) {
#ifdef __x86_64
        assert(cache);
        auto texture = cache->get(filename, error);
        if (!texture) {
                std::cerr << "No ptex texture: " << filename << ", " << error << std::endl;
                return;
        }
        texture->release();
#endif
}

void evaluate(const char *filename, long int faceIndex, float u, float v, float rgb[3]) {
#ifdef __x86_64
        auto texture = cache->get(filename, error);
        auto options(PtexFilter::FilterType::f_bspline);
        auto filter = PtexFilter::getFilter(texture, options);
        float dudx = 0.f;
        float dvdx = 0.f;
        float dudy = 0.f;
        float dvdy = 0.f;
        float result[3];
        filter->eval(result, 0, 3, faceIndex, u, v, dudx, dvdx, dudy, dvdy);
        rgb[0] = result[0];
        rgb[1] = result[1];
        rgb[2] = result[2];
        filter->release();
        texture->release();
#endif
}
}
