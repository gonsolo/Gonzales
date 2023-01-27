#include <OpenImageIO/texture.h>

OIIO::TextureSystem* textureSystem;

#ifdef __cplusplus
extern "C" {
#endif

void createTextureSystem() {
	textureSystem = OIIO::TextureSystem::create();
}

void destroyTextureSystem() {
	OIIO::ustring filename("bla.exr");
        OIIO::TextureOpt options;
        float s = 0.5;
        float t = 0.5;
        float dsdx = 0;
        float dtdx = 0;
        float dsdy = 0;
        float dtdy = 0;
        int nchannels = 3;
        float result[3];
        auto successful = textureSystem->texture(filename, options, s, t, dsdx, dtdx, dsdy, dtdy, nchannels, result);
        if (successful) {
                printf("result: %f %f %f\n", result[0], result[1], result[2]);
        } else {
                printf("Not successful.\n");
        }
}

void texture() {
	OIIO::TextureSystem::destroy(textureSystem);
}

#ifdef __cplusplus
}
#endif

