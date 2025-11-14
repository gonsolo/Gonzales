#include <OpenImageIO/imageio.h>
#include <OpenImageIO/texture.h>

std::shared_ptr<OIIO::TextureSystem> textureSystem;

#ifdef __cplusplus
extern "C" {
#endif

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

void writeImageInTiles(const char *filename_c, const float *pixels, const int xres, const int yres,
                       const int tileWidth, const int tileHeight) {
    
    const int channels = 4;
    std::unique_ptr<OIIO::ImageOutput> out = OIIO::ImageOutput::create(filename_c);
    if (!out) return;
    
    OIIO::ImageSpec spec(xres, yres, channels, OIIO::TypeDesc::FLOAT);
    spec.tile_width = tileWidth;
    spec.tile_height = tileHeight;
    
    if (!out->open(filename_c, spec)) {
        std::cerr << "ERROR: Konnte Datei nicht öffnen: " << out->geterror() << std::endl;
        return;
    }
    
    const ptrdiff_t channel_stride = sizeof(float);
    const ptrdiff_t x_stride = channels * channel_stride;
    const ptrdiff_t y_stride = (ptrdiff_t)xres * x_stride;
    
    int nxtiles = (xres + tileWidth - 1) / tileWidth;
    int nytiles = (yres + tileHeight - 1) / tileHeight;
    
    for (int ty = 0; ty < nytiles; ++ty) {
        for (int tx = 0; tx < nxtiles; ++tx) {
            int x_begin = tx * tileWidth;
            int y_begin = ty * tileHeight;
            ptrdiff_t index_offset = (ptrdiff_t)y_begin * xres + x_begin;
            const float *tile_ptr = pixels + index_offset * channels;

            OIIO::image_span<const float> tile_span( 
                tile_ptr, 
                (ptrdiff_t)channels, 
                (size_t)tileWidth, 
                (size_t)tileHeight, 
                1,                       // depth (z-dimension)
                channel_stride,          // channel stride
                x_stride,                // x stride
                y_stride                 // y stride
            );
            
            if (!out->write_tile(x_begin, y_begin, 0, tile_span)) {
                std::cerr << "ERROR: Fehler beim Schreiben der Kachel (" << tx << "," << ty
                          << "): " << out->geterror() << std::endl;
                goto cleanup;
            }
        }
    }

cleanup:
    out->close();
}

void writeImage(const char *filename_c, const float *pixels, const int xres, const int yres,
                const int tileWidth, const int tileHeight) {
        writeImageInTiles(filename_c, pixels, xres, yres, tileWidth, tileHeight);
        // const int channels = 4; // RGBA
        // std::unique_ptr<OIIO::ImageOutput> out = OIIO::ImageOutput::create(filename_c);
        // if (!out)
        //         return;
        // OIIO::ImageSpec spec(xres, yres, channels, OIIO::TypeDesc::FLOAT);
        // spec.tile_width = tileWidth;
        // spec.tile_height = tileHeight;
        // out->open(filename_c, spec);
        // out->write_image(OIIO::TypeDesc::FLOAT, pixels);
        // out->close();
}

#ifdef __cplusplus
}
#endif
