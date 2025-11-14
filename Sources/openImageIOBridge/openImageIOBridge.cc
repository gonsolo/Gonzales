#include <OpenImageIO/imageio.h>
#include <OpenImageIO/texture.h>
#include <iostream>
#include <memory>
#include <algorithm>

std::shared_ptr<OIIO::TextureSystem> textureSystem;

// --- Helper function definitions (kept outside extern "C") ---

// writeTile implementation is unchanged and does not need C linkage since it's only called internally by writeTiledImageBody
bool writeTile(OIIO::ImageOutput *out, const float *pixels, int xres, int channels, int tx, int ty,
               int tileWidth, int tileHeight, ptrdiff_t channel_stride, ptrdiff_t x_stride,
               ptrdiff_t y_stride) {

        int x_begin = tx * tileWidth;
        int y_begin = ty * tileHeight;

        ptrdiff_t index_offset = (ptrdiff_t)y_begin * xres + x_begin;
        const float *tile_ptr = pixels + index_offset * channels;

        OIIO::image_span<const float> tile_span(tile_ptr, (ptrdiff_t)channels, (size_t)tileWidth,
                                                (size_t)tileHeight, 1, channel_stride, x_stride, y_stride);

        if (!out->write_tile(x_begin, y_begin, 0, tile_span)) {
                std::cerr << "ERROR: Failed to write tile (" << tx << "," << ty << "): " << out->geterror()
                          << std::endl;
                return false;
        }
        return true;
}


// --- Exposed C functions (Must be compiled with C linkage) ---
#ifdef __cplusplus
extern "C" {
#endif

// The implementation is now inside extern "C"
OIIO::ImageOutput *openImageForTiledWriting(const char *filename_c, int xres, int yres,
                                            int tileWidth, int tileHeight, int channels) {

        std::unique_ptr<OIIO::ImageOutput> out_uptr = OIIO::ImageOutput::create(filename_c);
        
        if (!out_uptr)
                return nullptr;

        OIIO::ImageSpec spec(xres, yres, channels, OIIO::TypeDesc::FLOAT);
        spec.tile_width = tileWidth;
        spec.tile_height = tileHeight;

        if (!out_uptr->open(filename_c, spec)) {
                std::cerr << "ERROR: Could not open file: " << out_uptr->geterror() << std::endl;
                return nullptr;
        }

        return out_uptr.release();
}

// The implementation is now inside extern "C"
bool writeTiledImageBody(OIIO::ImageOutput *out, const float *pixels, int xres, int yres, int tileWidth,
                         int tileHeight, int channels) {

        const ptrdiff_t channel_stride = sizeof(float);
        const ptrdiff_t x_stride = channels * channel_stride;
        const ptrdiff_t y_stride = (ptrdiff_t)xres * x_stride;

        int nxtiles = (xres + tileWidth - 1) / tileWidth;
        int nytiles = (yres + tileHeight - 1) / tileHeight;

        for (int ty = 0; ty < nytiles; ++ty) {
                for (int tx = 0; tx < nxtiles; ++tx) {

                        if (!writeTile(out, pixels, xres, channels, tx, ty, tileWidth, tileHeight,
                                       channel_stride, x_stride, y_stride)) {
                                return false;
                        }
                }
        }

        return true;
}

void closeImageOutput(OIIO::ImageOutput* out) {
    if (out) {
        out->close();
        delete out;
    }
}

void createTextureSystem() { textureSystem = OIIO::TextureSystem::create(); }

void destroyTextureSystem() { OIIO::TextureSystem::destroy(textureSystem); }

bool texture(const char *filename_c, float s, float t, float result[3]) {
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
