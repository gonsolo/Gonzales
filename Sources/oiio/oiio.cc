#include <OpenImageIO/texture.h>

OIIO::TextureSystem* textureSystem;

#ifdef __cplusplus
extern "C" {
#endif

void createTextureSystem() {
	textureSystem = OIIO::TextureSystem::create();
}

void destroyTextureSystem() {
	OIIO::TextureSystem::destroy(textureSystem);
}

void texture(const char* filename_c, float s, float t) {
	OIIO::ustring filename(filename_c);
        OIIO::TextureOpt options;
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

#ifdef __cplusplus
}
#endif

