#include <OpenImageIO/imageio.h>
#include <OpenImageIO/imagecache.h>
#include <OpenImageIO/texture.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

std::shared_ptr<OIIO::TextureSystem> textureSystem;

// sRGB → linear (same conversion as CPU shader's _srgb_to_linear)
static inline float srgb_to_linear(float c) {
        return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

// Built through OIIO's own uint8→float conversion so results are bit-identical
// to reading the image as FLOAT and converting every texel.
static std::array<float, 256> make_uint8_lut(bool decode) {
        std::array<unsigned char, 256> in;
        for (int i = 0; i < 256; ++i)
                in[i] = static_cast<unsigned char>(i);
        std::array<float, 256> lut;
        OIIO::convert_pixel_values(OIIO::TypeDesc::UINT8, in.data(), OIIO::TypeDesc::FLOAT, lut.data(), 256);
        if (decode)
                for (float &v : lut)
                        v = srgb_to_linear(v);
        return lut;
}

template <typename T, typename Map>
static void to_linear_rgb(const T *src, int64_t n, int nc, float *dst, Map map) {
        for (int64_t i = 0; i < n; ++i) {
                const T *p = src + i * nc;
                const float r = nc > 0 ? map(p[0]) : 0.0f;
                dst[i * 3 + 0] = r;
                dst[i * 3 + 1] = nc > 1 ? map(p[1]) : r;
                dst[i * 3 + 2] = nc > 2 ? map(p[2]) : r;
        }
}

// --- Exposed C functions (Must be compiled with C linkage) ---
#ifdef __cplusplus
extern "C" {
#endif

// Returns an OIIO::ImageOutput* (void* in Swift)
OIIO::ImageOutput *openImageForTiledWriting(const char *filename_c, int xres, int yres, int tileWidth,
                                            int tileHeight, int channels, int fullWidth, int fullHeight,
                                            int x, int y) {

        std::unique_ptr<OIIO::ImageOutput> out_uptr = OIIO::ImageOutput::create(filename_c);

        if (!out_uptr)
                return nullptr;

        OIIO::ImageSpec spec(xres, yres, channels, OIIO::TypeDesc::FLOAT);
        spec.tile_width = tileWidth;
        spec.tile_height = tileHeight;
        spec.full_x = 0;
        spec.full_y = 0;
        spec.full_width = fullWidth;
        spec.full_height = fullHeight;
        spec.x = x;
        spec.y = y;

        if (!out_uptr->open(filename_c, spec)) {
                std::cerr << "ERROR: Could not open file: " << out_uptr->geterror() << std::endl;
                return nullptr;
        }

        return out_uptr.release();
}

// Function to write a single tile (called inside the Swift loop)
bool writeSingleTile(OIIO::ImageOutput *out, const float *pixels, int xres, int channels, int tx, int ty,
                     int xOffset, int yOffset, int tileWidth, int tileHeight, ptrdiff_t channel_stride,
                     ptrdiff_t x_stride, ptrdiff_t y_stride) {

        int buffer_x = tx * tileWidth;
        int buffer_y = ty * tileHeight;

        ptrdiff_t index_offset = (ptrdiff_t)buffer_y * xres + buffer_x;
        const float *tile_ptr = pixels + index_offset * channels;

        OIIO::image_span<const float> tile_span(tile_ptr, (ptrdiff_t)channels, (size_t)tileWidth,
                                                (size_t)tileHeight, 1, channel_stride, x_stride, y_stride);

        int target_x = buffer_x + xOffset;
        int target_y = buffer_y + yOffset;

        if (!out->write_tile(target_x, target_y, 0, tile_span)) {
                std::cerr << "ERROR: Failed to write tile (" << target_x << "," << target_y
                          << "): " << out->geterror() << std::endl;
                return false;
        }
        return true;
}

void closeImageOutput(OIIO::ImageOutput *out) {
        if (out) {
                out->close();
                delete out;
        }
}

// Existing texture system functions (kept for completeness)
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
        bool ok = textureSystem->texture(filename, options, s, t, dsdx, dtdx, dsdy, dtdy, nchannels, result);
        // TextureOpt::fill defaults to 0, so a single-channel (greyscale) or
        // two-channel source image sampled here for 3 channels comes back
        // with the missing channel(s) at 0 instead of replicated luminance --
        // e.g. a grey texture turns pure red (R=lum, G=B=0). Detect and
        // replicate, matching load_texture_rgb's already-correct nc-aware
        // handling used by the GPU upload path.
        if (ok) {
                const OIIO::ImageSpec *spec = textureSystem->imagecache()->imagespec(filename);
                if (spec) {
                        int nc = spec->nchannels;
                        if (nc <= 1) {
                                result[1] = result[0];
                                result[2] = result[0];
                        } else if (nc == 2) {
                                result[2] = result[0];
                        }
                }
        }
        return ok;
}

int load_texture_rgb(const char *filename, float **data, int *width, int *height, int raw) {
        auto in = OIIO::ImageInput::open(filename);
        if (!in) return 0;
        const OIIO::ImageSpec &spec = in->spec();
        *width  = spec.width;
        *height = spec.height;
        const int64_t n = int64_t(spec.width) * spec.height;
        const int nc = spec.nchannels;
        const bool hdr = strstr(filename, ".exr") != nullptr || strstr(filename, ".hdr") != nullptr || strstr(filename, ".pfm") != nullptr;
        // raw != 0 keeps linear data (e.g. normal maps); only sRGB colour textures get decoded.
        const bool decode = !hdr && !raw;
        *data = (float *)malloc(n * 3 * sizeof(float));
        if (!*data) return 0;
        bool ok;
        if (spec.format == OIIO::TypeDesc::UINT8 && spec.channelformats.empty()) {
                static const std::array<float, 256> lut_linear = make_uint8_lut(false);
                static const std::array<float, 256> lut_srgb = make_uint8_lut(true);
                const std::array<float, 256> &lut = decode ? lut_srgb : lut_linear;
                std::unique_ptr<unsigned char[]> buf(new unsigned char[n * nc]);
                ok = in->read_image(0, 0, 0, nc, OIIO::TypeDesc::UINT8, buf.get());
                if (ok)
                        to_linear_rgb(buf.get(), n, nc, *data, [&](unsigned char v) { return lut[v]; });
        } else {
                std::unique_ptr<float[]> buf(new float[n * nc]);
                ok = in->read_image(0, 0, 0, nc, OIIO::TypeDesc::FLOAT, buf.get());
                if (ok)
                        to_linear_rgb(buf.get(), n, nc, *data, [&](float v) { return decode ? srgb_to_linear(v) : v; });
        }
        in->close();
        // The buffer is uninitialized now, so a failed read must not be handed back as a texture.
        if (!ok) {
                free(*data);
                *data = nullptr;
                return 0;
        }
        return 1;
}

int free_texture_rgb(float *data) {
        free(data);
        return 0;
}

static bool is_hdr_ext(const char *filename) {
        const char *dot = strrchr(filename, '.');
        if (!dot) return true;
        return strcmp(dot, ".exr") == 0 || strcmp(dot, ".hdr") == 0 || strcmp(dot, ".pfm") == 0;
}

struct RGB {
        float r, g, b;

        static RGB from(const float *p) { return {p[0], p[1], p[2]}; }

        float luminance() const { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

        RGB reinhard() const {
                float lum = luminance();
                float scale = lum > 1e-6f ? (lum / (1.0f + lum)) / lum : 1.0f;
                return {r * scale, g * scale, b * scale};
        }

        RGB to_srgb() const { return {linear_to_srgb(r), linear_to_srgb(g), linear_to_srgb(b)}; }

        uint8_t to_u8(float ch) const { return (uint8_t)std::min(255.0f, ch * 255.0f + 0.5f); }
        void store_u8(uint8_t *p) const { p[0] = to_u8(r); p[1] = to_u8(g); p[2] = to_u8(b); }

private:
        static float linear_to_srgb(float x) {
                if (x <= 0.0f) return 0.0f;
                if (x >= 1.0f) return 1.0f;
                return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
        }
};

// Write a float RGB buffer.  EXR/HDR → float32 tiles.
// PNG/JPG/etc. → Reinhard-on-luminance tonemap + sRGB gamma → uint8.
// Returns 1 on success, 0 on failure.
int write_image_rgb(const char *filename, const float *rgb, int width, int height,
                    int tile_w, int tile_h) {
        auto out = OIIO::ImageOutput::create(filename);
        if (!out) {
                std::cerr << "write_image_rgb: cannot create writer for " << filename << std::endl;
                return 0;
        }
        if (is_hdr_ext(filename)) {
                OIIO::ImageSpec spec(width, height, 3, OIIO::TypeDesc::FLOAT);
                spec.tile_width = tile_w;
                spec.tile_height = tile_h;
                if (!out->open(filename, spec)) return 0;
                out->write_image(OIIO::TypeDesc::FLOAT, rgb);
        } else {
                int n = width * height;
                std::vector<uint8_t> ldr(n * 3);
                for (int i = 0; i < n; ++i)
                        RGB::from(rgb + i * 3).reinhard().to_srgb().store_u8(ldr.data() + i * 3);
                OIIO::ImageSpec spec(width, height, 3, OIIO::TypeDesc::UINT8);
                if (!out->open(filename, spec)) return 0;
                out->write_image(OIIO::TypeDesc::UINT8, ldr.data());
        }
        out->close();
        return 1;
}

// See oiio.h for the dataWindow/displayWindow contract.
int write_image_rgb_windowed(const char *filename, const float *rgb, int width, int height,
                             int full_width, int full_height, int x, int y,
                             int tile_w, int tile_h) {
        auto out = OIIO::ImageOutput::create(filename);
        if (!out) {
                std::cerr << "write_image_rgb_windowed: cannot create writer for " << filename << std::endl;
                return 0;
        }
        if (is_hdr_ext(filename)) {
                OIIO::ImageSpec spec(width, height, 3, OIIO::TypeDesc::FLOAT);
                spec.tile_width = tile_w;
                spec.tile_height = tile_h;
                spec.full_x = 0;
                spec.full_y = 0;
                spec.full_width = full_width;
                spec.full_height = full_height;
                spec.x = x;
                spec.y = y;
                if (!out->open(filename, spec)) return 0;
                out->write_image(OIIO::TypeDesc::FLOAT, rgb);
        } else {
                int n = width * height;
                std::vector<uint8_t> ldr(n * 3);
                for (int i = 0; i < n; ++i)
                        RGB::from(rgb + i * 3).reinhard().to_srgb().store_u8(ldr.data() + i * 3);
                OIIO::ImageSpec spec(width, height, 3, OIIO::TypeDesc::UINT8);
                if (!out->open(filename, spec)) return 0;
                out->write_image(OIIO::TypeDesc::UINT8, ldr.data());
        }
        out->close();
        return 1;
}

#ifdef __cplusplus
}
#endif
