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

bool texture(const char* filename_c, float s, float t, float result[3]) {
	OIIO::ustring filename(filename_c);
        OIIO::TextureOpt options;
        float dsdx = 0;
        float dtdx = 0;
        float dsdy = 0;
        float dtdy = 0;
        int nchannels = 3;
        return textureSystem->texture(filename, options, s, t, dsdx, dtdx, dsdy, dtdy, nchannels, result);
}

#ifdef __cplusplus
}
#endif

